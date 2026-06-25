"""Gradio web demo for ``aw-generate``.

Launches a small browser UI that wraps :func:`generate`, exposing the
most useful controls (text prompt, visual-cube shape, sampler steps,
CFG scale, seed, model preset). The demo runs against random weights
by default but accepts ``--dynamics-ckpt`` / ``--decoder-ckpt`` to load
trained models.

The Gradio import is intentionally lazy so the rest of the codebase
remains importable without the dependency.

Examples
--------
::

    pip install gradio
    aw-demo --device cpu --vocab tiny --preset toy

    aw-demo --device cuda --vocab default --preset b7 \
        --dynamics-ckpt outputs/ckpts/dynamics/step-00050000 \
        --decoder-ckpt outputs/ckpts/dit/step-00010000 \
        --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from another_world.inference.generation import GenerationConfig, generate
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


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _maybe_load(model: torch.nn.Module, ckpt_dir: Path | None, name: str) -> None:
    if ckpt_dir is None:
        _LOG.info("%s: no checkpoint, using random weights", name)
        return
    meta = load_checkpoint(ckpt_dir, model=model, strict=False)
    _LOG.info("%s: loaded checkpoint step=%d", name, meta.step)


def _build_models(args: argparse.Namespace) -> tuple[
    VocabLayout, MultimodalDynamicsModel, DiTDecoder
]:
    layout = (
        VocabLayout.tiny() if args.vocab == "tiny" else VocabLayout.default()
    )
    device = _resolve_device(args.device)
    dyn = MultimodalDynamicsModel(
        _DYN_PRESETS[args.preset](vocab_size=layout.total_size)
    ).to(device).eval()
    _maybe_load(dyn, args.dynamics_ckpt, "dynamics")
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
    return layout, dyn, dec


def _pixels_to_preview(pixels: torch.Tensor) -> Any:
    """Convert ``[1, C, T, H, W]`` -> numpy array suitable for Gradio.

    If C == 3 we treat the tensor as RGB and return a uint8 ``[T, H, W, 3]``;
    otherwise we average channels into grayscale for a debug preview.
    """

    import numpy as np

    if pixels.dim() != 5:
        raise ValueError(f"expected 5-D pixels, got {tuple(pixels.shape)}")
    arr = pixels[0].detach().cpu().to(torch.float32)
    if arr.min() < -0.01:
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 1.0001:
        arr = arr * 255.0
    arr = arr.clamp(0, 255)
    if arr.shape[0] >= 3:
        arr = arr[:3].permute(1, 2, 3, 0)        # [T, H, W, 3]
    else:
        arr = arr.mean(dim=0).unsqueeze(-1).repeat(1, 1, 1, 3)
    return arr.to(torch.uint8).numpy()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aw-demo", description="Gradio web demo wrapping aw-generate.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--vocab", default="tiny", choices=["tiny", "default"])
    p.add_argument("--preset", default="toy", choices=list(_DYN_PRESETS))
    p.add_argument("--decoder-dim", type=int, default=64)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--decoder-heads", type=int, default=4)
    p.add_argument("--decoder-patch", type=int, default=2)
    p.add_argument("--decoder-patch-t", type=int, default=1)
    p.add_argument("--decoder-channels", type=int, default=4)
    p.add_argument("--dynamics-ckpt", type=Path, default=None)
    p.add_argument("--decoder-ckpt", type=Path, default=None)
    p.add_argument("--text-tokenizer", default="hash",
                   choices=["hash", "whitespace", "hf"])
    p.add_argument("--hf-text-model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true",
                   help="enable a temporary public URL via Gradio share")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError:
        print(
            "gradio is not installed. `pip install gradio` and retry.",
            file=sys.stderr,
        )
        return 1

    layout, dyn, dec = _build_models(args)
    text_tk = build_text_tokenizer(
        kind=args.text_tokenizer,
        vocab_size=layout.text_size,
        hf_model=args.hf_text_model,
        max_len=32,
    )

    def _infer(
        prompt: str,
        visual_frames: int,
        visual_h: int,
        visual_w: int,
        pixel_t: int,
        pixel_h: int,
        pixel_w: int,
        sampler: str,
        steps: int,
        cfg_scale: float,
        temperature: float,
        seed: int,
    ):
        text_ids = (
            [int(i) % layout.text_size for i in text_tk.encode(prompt).tolist()]
            if prompt else None
        )
        cfg = GenerationConfig(
            visual_frames=int(visual_frames),
            visual_height=int(visual_h),
            visual_width=int(visual_w),
            sampler=sampler,
            sampler_steps=int(steps),
            cfg_scale=float(cfg_scale),
            temperature=float(temperature),
            latent_channels=args.decoder_channels,
            pixel_t=int(pixel_t),
            pixel_h=int(pixel_h),
            pixel_w=int(pixel_w),
            seed=int(seed),
        )
        result = generate(
            dynamics=dyn, decoder=dec,
            text_ids=text_ids, layout=layout, config=cfg,
        )
        preview = _pixels_to_preview(result.pixels)
        tokens_info = (
            f"visual tokens: {tuple(result.visual_tokens.shape)}, "
            f"pixels: {tuple(result.pixels.shape)}"
        )
        return preview, tokens_info

    with gr.Blocks(title="Another World - Generation Demo") as demo:
        gr.Markdown("# Another World - Generation Demo")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="a cat on a windowsill")
                with gr.Row():
                    visual_frames = gr.Slider(1, 16, 2, step=1, label="visual frames T'")
                    visual_h = gr.Slider(2, 32, 4, step=1, label="visual H'")
                    visual_w = gr.Slider(2, 32, 4, step=1, label="visual W'")
                with gr.Row():
                    pixel_t = gr.Slider(1, 64, 2, step=1, label="pixel T")
                    pixel_h = gr.Slider(8, 512, 16, step=8, label="pixel H")
                    pixel_w = gr.Slider(8, 512, 16, step=8, label="pixel W")
                with gr.Row():
                    sampler = gr.Dropdown(
                        choices=["euler", "dpm_solver"], value="euler",
                        label="sampler",
                    )
                    steps = gr.Slider(1, 100, 8, step=1, label="sampler steps")
                    cfg_scale = gr.Slider(1.0, 10.0, 1.0, step=0.1, label="CFG scale")
                with gr.Row():
                    temperature = gr.Slider(0.1, 2.0, 1.0, step=0.05, label="temperature")
                    seed = gr.Number(0, label="seed", precision=0)
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                preview = gr.Gallery(label="Frames", show_label=True)
                info = gr.Markdown()

        def _wrap(*args):
            preview_arr, msg = _infer(*args)
            frames = [preview_arr[i] for i in range(preview_arr.shape[0])]
            return frames, msg

        go.click(
            _wrap,
            inputs=[
                prompt, visual_frames, visual_h, visual_w,
                pixel_t, pixel_h, pixel_w,
                sampler, steps, cfg_scale, temperature, seed,
            ],
            outputs=[preview, info],
        )

    _LOG.info(
        "Launching Gradio on %s:%d (share=%s)", args.host, args.port, args.share,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
