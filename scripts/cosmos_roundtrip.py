#!/usr/bin/env python
"""End-to-end Cosmos-Tokenizer demo: video file -> tokens -> reconstructed video.

This script is intentionally thin so it can serve as a recipe for the data
pipeline in stage 1.3.

Example
-------
First download a checkpoint (one-time, requires HF auth + GPU box):

    python -c "from another_world.tokenizers.visual.cosmos \
        import download_cosmos_checkpoint as d; \
        d('Cosmos-1.0-Tokenizer-DV8x16x16')"

Then encode + decode a video:

    python scripts/cosmos_roundtrip.py \
        --model Cosmos-1.0-Tokenizer-DV8x16x16 \
        --ckpt-dir .cache/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16 \
        --input  test_data/video.mp4 \
        --output outputs/reconstructed.mp4 \
        --frames 17 --height 256 --width 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from another_world.tokenizers.visual.cosmos import (
    DEFAULT_VIDEO_MODEL,
    CosmosVideoTokenizer,
)
from another_world.utils.logging import get_logger

_LOG = get_logger("cosmos_roundtrip")


def _load_video(path: Path, frames: int, height: int, width: int) -> torch.Tensor:
    """Load and pre-process a video into a Cosmos-friendly tensor.

    Returns ``[1, 3, T, H, W]`` in float32, values in [-1, 1].
    """

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pyav is required to load videos. Install with `pip install av`."
        ) from exc
    import numpy as np  # local import keeps numpy optional for non-video paths

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    pictures: list[np.ndarray] = []
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")
        pictures.append(img)
        if len(pictures) >= frames:
            break
    if len(pictures) < frames:
        raise ValueError(
            f"video {path} has only {len(pictures)} decoded frames, "
            f"need {frames}."
        )
    container.close()

    arr = np.stack(pictures, axis=0)  # [T, H, W, 3]
    tensor = torch.from_numpy(arr).float() / 127.5 - 1.0  # -> [-1, 1]
    tensor = tensor.permute(3, 0, 1, 2)  # [3, T, H, W]
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),  # [1, 3, T, H, W]
        size=(frames, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return tensor


def _save_video(tensor: torch.Tensor, path: Path, fps: int = 24) -> None:
    """Save a ``[1, 3, T, H, W]`` tensor (values ~[-1,1]) as mp4."""

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pyav is required to save videos.") from exc
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().to(torch.float32).clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    _, _, frames, height, width = arr.shape
    arr = arr[0].transpose(1, 2, 3, 0)  # [T, H, W, 3]

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(frames):
        frame = av.VideoFrame.from_ndarray(arr[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cosmos-Tokenizer roundtrip demo")
    p.add_argument("--model", default=DEFAULT_VIDEO_MODEL)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args(argv)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    tk = CosmosVideoTokenizer.from_local(
        args.model,
        args.ckpt_dir,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )
    expected_frames = tk.round_frames(args.frames)
    if expected_frames != args.frames:
        _LOG.warning(
            "Requested %d frames; Cosmos requires 1 + k*%d; using %d.",
            args.frames, tk.spec.temporal_compression, expected_frames,
        )

    _LOG.info("Loading %s -> [%dx3x%dx%dx%d]",
              args.input, 1, expected_frames, args.height, args.width)
    video = _load_video(args.input, expected_frames, args.height, args.width)

    _LOG.info("Encoding with %s ...", args.model)
    encoded = tk.encode(video)
    _LOG.info(
        "Encoded outputs: %s",
        [t.shape if hasattr(t, "shape") else type(t).__name__ for t in encoded],
    )

    _LOG.info("Decoding ...")
    recon = tk.decode(encoded[0])

    _LOG.info("Saving reconstruction -> %s", args.output)
    _save_video(recon, args.output, fps=args.fps)

    diff = (recon.float().cpu() - video.float()).abs().mean().item()
    _LOG.info("Mean abs reconstruction error: %.4f", diff)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
