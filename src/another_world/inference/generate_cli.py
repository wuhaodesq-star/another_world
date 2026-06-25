"""``aw-generate`` CLI.

End-to-end video generation entry point. Loads a dynamics-model and DiT
checkpoint pair, runs token rollout + DiT decode, and writes the result
to disk as either:

- ``.pt``  raw pixel-latent tensor (default; cheapest, no extra deps)
- ``.npy`` numpy array (for inspection in non-torch tooling)
- ``.mp4`` video file (requires ``pyav`` and an image-decoded tensor;
  if the model outputs latents rather than pixels we still emit a
  best-effort ``.pt`` so the user can post-process)

Examples
--------
Pure-CPU smoke (random weights, useful for verifying the pipeline)::

    aw-generate --device cpu --steps 4 \
        --vocab tiny --preset toy \
        --visual-frames 1 --visual-h 2 --visual-w 2 \
        --pixel-t 2 --pixel-h 16 --pixel-w 16 \
        --out outputs/gen-smoke.pt

Resume from real checkpoints::

    aw-generate \
        --dynamics-ckpt outputs/ckpts/dynamics/step-00050000 \
        --decoder-ckpt outputs/ckpts/dit/step-00010000 \
        --text "a cat sitting on a windowsill" \
        --steps 30 --sampler euler \
        --visual-frames 2 --visual-h 16 --visual-w 16 \
        --pixel-t 17 --pixel-h 256 --pixel-w 256 \
        --out outputs/gen.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from another_world.inference.generation import (
    GenerationConfig,
    generate,
)
from another_world.models.decoder import DiTDecoder, DiTDecoderConfig
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.text import build_text_tokenizer
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.checkpoint import load_checkpoint
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


_DYN_PRESETS = {
    "toy": MultimodalDynamicsConfig.toy,
    "m350": MultimodalDynamicsConfig.m350,
    "b1": MultimodalDynamicsConfig.b1,
    "b3": MultimodalDynamicsConfig.b3,
    "b7": MultimodalDynamicsConfig.b7,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aw-generate",
        description="Generate a video from text and/or a first-frame token cube.",
    )
    # vocab + model
    p.add_argument("--vocab", default="tiny", choices=["tiny", "default"])
    p.add_argument("--preset", default="toy", choices=list(_DYN_PRESETS))
    p.add_argument("--decoder-dim", type=int, default=64)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--decoder-heads", type=int, default=4)
    p.add_argument("--decoder-patch", type=int, default=2)
    p.add_argument("--decoder-patch-t", type=int, default=1,
                   help="temporal patch (>1 enables 3-D spatiotemporal patching)")
    p.add_argument("--decoder-channels", type=int, default=4)
    # checkpoints
    p.add_argument("--dynamics-ckpt", type=Path, default=None,
                   help="checkpoint directory for the dynamics model")
    p.add_argument("--decoder-ckpt", type=Path, default=None,
                   help="checkpoint directory for the DiT decoder")
    # prompt
    p.add_argument("--text", default=None,
                   help="natural-language text prompt")
    p.add_argument("--text-ids", default=None,
                   help="comma-separated explicit token ids; overrides --text")
    p.add_argument("--text-tokenizer", default="hash",
                   choices=["hash", "whitespace", "hf"],
                   help="text tokenizer kind (hash is offline + dependency-free)")
    p.add_argument("--hf-text-model", default="meta-llama/Meta-Llama-3-8B",
                   help="HF model id when --text-tokenizer=hf")
    p.add_argument("--first-frame", type=Path, default=None,
                   help="path to a .pt tensor of shape [T_prefix, H', W'] (or "
                        "[H', W']) with local visual ids; rollout conditions "
                        "on these.")
    p.add_argument("--action-ids", default=None,
                   help="comma-separated local action ids for action-conditioned "
                        "rollout")
    # rollout
    p.add_argument("--visual-frames", type=int, default=2)
    p.add_argument("--visual-h", type=int, default=4)
    p.add_argument("--visual-w", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-kv-cache", action="store_true",
                   help="disable the KV cache and use the O(T^2) recompute path")
    # sampler
    p.add_argument("--sampler", default="euler", choices=["euler", "dpm_solver"])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--pixel-t", type=int, default=2)
    p.add_argument("--pixel-h", type=int, default=16)
    p.add_argument("--pixel-w", type=int, default=16)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="classifier-free guidance scale (>1 enables CFG)")
    p.add_argument("--null-token-id", type=int, default=None,
                   help="visual token id used for the unconditional branch")
    # misc
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _make_text_ids(args: argparse.Namespace, layout: VocabLayout) -> list[int] | None:
    if args.text_ids:
        return [int(x) for x in args.text_ids.split(",") if x.strip()]
    if args.text:
        tokenizer = build_text_tokenizer(
            kind=args.text_tokenizer,
            vocab_size=layout.text_size,
            hf_model=args.hf_text_model,
            max_len=32,
        )
        ids = tokenizer.encode(args.text).tolist()
        # Clamp into the local text slab range so encode_text() won't reject.
        return [int(i) % layout.text_size for i in ids]
    return None


def _save_output(pixels: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        torch.save(pixels.cpu(), path)
        _LOG.info("Saved %s (%s, %s)", path, tuple(pixels.shape), pixels.dtype)
        return
    if suffix == ".npy":
        import numpy as np

        np.save(path, pixels.cpu().numpy())
        _LOG.info("Saved %s (%s)", path, tuple(pixels.shape))
        return
    if suffix == ".mp4":
        _save_mp4(pixels, path)
        return
    # default: torch.save with .pt extension auto-appended
    torch.save(pixels.cpu(), path.with_suffix(".pt"))
    _LOG.info("Unknown suffix %r; saved as .pt instead", suffix)


def _save_mp4(pixels: torch.Tensor, path: Path) -> None:
    """Best-effort mp4 export.

    Expects a tensor we can interpret as ``[1, 3, T, H, W]`` in ``[-1, 1]``
    or ``[0, 1]``; for latent tensors (C != 3) we fall back to ``.pt``.
    """

    if pixels.dim() != 5 or pixels.shape[1] != 3:
        fallback = path.with_suffix(".pt")
        torch.save(pixels.cpu(), fallback)
        _LOG.warning(
            "mp4 export skipped (need [1, 3, T, H, W] but got %s); wrote %s",
            tuple(pixels.shape), fallback,
        )
        return

    try:
        import av  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised when av missing
        fallback = path.with_suffix(".pt")
        torch.save(pixels.cpu(), fallback)
        _LOG.warning("pyav not installed (%s); wrote %s", exc, fallback)
        return

    arr = pixels[0].detach().cpu().to(torch.float32)
    if arr.min() < -0.01:
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 1.0001:
        arr = arr * 255.0
    arr = arr.clamp(0, 255).to(torch.uint8).numpy()
    _, t, h, w = arr.shape
    arr = arr.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=24)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(t):
        frame = av.VideoFrame.from_ndarray(arr[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    _LOG.info("Saved mp4 -> %s", path)


def _maybe_load(model: torch.nn.Module, ckpt_dir: Path | None, name: str) -> None:
    if ckpt_dir is None:
        _LOG.info("%s: no checkpoint, using random weights", name)
        return
    meta = load_checkpoint(ckpt_dir, model=model, strict=False)
    _LOG.info("%s: loaded checkpoint step=%d", name, meta.step)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    layout = VocabLayout.tiny() if args.vocab == "tiny" else VocabLayout.default()
    _LOG.info("vocab layout total=%d", layout.total_size)

    # ----- dynamics -----
    dyn_cfg = _DYN_PRESETS[args.preset](vocab_size=layout.total_size)
    dyn = MultimodalDynamicsModel(dyn_cfg).to(device).eval()
    _maybe_load(dyn, args.dynamics_ckpt, "dynamics")

    # ----- decoder -----
    dec_cfg = DiTDecoderConfig(
        in_channels=args.decoder_channels,
        out_channels=args.decoder_channels,
        patch_size=args.decoder_patch,
        patch_t=args.decoder_patch_t,
        dim=args.decoder_dim,
        n_layers=args.decoder_layers,
        n_heads=args.decoder_heads,
        vocab_size=layout.visual_size,
    )
    dec = DiTDecoder(dec_cfg).to(device).eval()
    _maybe_load(dec, args.decoder_ckpt, "decoder")

    text_ids = _make_text_ids(args, layout)
    if text_ids is not None:
        _LOG.info("prompt token ids (%d): %s",
                  len(text_ids), text_ids[:16] + (["..."] if len(text_ids) > 16 else []))

    cfg = GenerationConfig(
        visual_frames=args.visual_frames,
        visual_height=args.visual_h,
        visual_width=args.visual_w,
        temperature=args.temperature,
        top_k=args.top_k,
        sampler=args.sampler,
        sampler_steps=args.steps,
        latent_channels=args.decoder_channels,
        pixel_t=args.pixel_t,
        pixel_h=args.pixel_h,
        pixel_w=args.pixel_w,
        seed=args.seed,
        use_kv_cache=not args.no_kv_cache,
        cfg_scale=args.cfg_scale,
        null_token_id=args.null_token_id,
    )
    # First-frame visual prefix (optional).
    first_frame = None
    if args.first_frame is not None:
        first_frame = torch.load(
            args.first_frame, map_location="cpu", weights_only=False,
        )
        if not isinstance(first_frame, torch.Tensor):
            raise SystemExit(
                f"--first-frame must point at a .pt tensor; got {type(first_frame)}"
            )
        _LOG.info("loaded first_frame: %s", tuple(first_frame.shape))

    action_ids = None
    if args.action_ids:
        action_ids = [int(x) for x in args.action_ids.split(",") if x.strip()]
        _LOG.info("action prefix ids: %s", action_ids)

    result = generate(
        dynamics=dyn, decoder=dec, text_ids=text_ids,
        layout=layout, config=cfg,
        first_frame=first_frame, action_ids=action_ids,
    )
    _LOG.info(
        "visual tokens: %s, pixels: %s",
        tuple(result.visual_tokens.shape),
        tuple(result.pixels.shape),
    )
    _save_output(result.pixels, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
