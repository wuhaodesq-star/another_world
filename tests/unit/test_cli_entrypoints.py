"""Tests for the CLI entry points (aw-generate, aw-eval)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from another_world.eval.cli import main as eval_main
from another_world.inference.generate_cli import main as generate_main


def test_aw_generate_writes_pt(tmp_path: Path) -> None:
    out = tmp_path / "result.pt"
    rc = generate_main([
        "--device", "cpu",
        "--vocab", "tiny",
        "--preset", "toy",
        "--visual-frames", "1",
        "--visual-h", "2",
        "--visual-w", "2",
        "--steps", "2",
        "--pixel-t", "2",
        "--pixel-h", "8",
        "--pixel-w", "8",
        "--decoder-dim", "32",
        "--decoder-layers", "2",
        "--decoder-heads", "4",
        "--decoder-patch", "2",
        "--decoder-channels", "4",
        "--out", str(out),
        "--seed", "0",
    ])
    assert rc == 0
    assert out.exists()
    pixels = torch.load(out, map_location="cpu", weights_only=False)
    assert pixels.shape == (1, 4, 2, 8, 8)


def test_aw_generate_with_text_prompt(tmp_path: Path) -> None:
    out = tmp_path / "result.pt"
    rc = generate_main([
        "--device", "cpu",
        "--vocab", "tiny",
        "--preset", "toy",
        "--visual-frames", "1",
        "--visual-h", "2",
        "--visual-w", "2",
        "--steps", "2",
        "--pixel-t", "2",
        "--pixel-h", "8",
        "--pixel-w", "8",
        "--decoder-dim", "32",
        "--decoder-layers", "2",
        "--decoder-heads", "4",
        "--decoder-patch", "2",
        "--decoder-channels", "4",
        "--text", "hello world",
        "--out", str(out),
        "--seed", "1",
    ])
    assert rc == 0
    assert out.exists()


def test_aw_generate_with_reference_image(tmp_path: Path) -> None:
    """--image should be tokenised by the mock visual tokenizer and used as prefix."""
    from PIL import Image
    import numpy as np

    image = tmp_path / "ref.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 1] = 255
    Image.fromarray(arr).save(image)

    out = tmp_path / "result.pt"
    rc = generate_main([
        "--device", "cpu",
        "--vocab", "tiny",
        "--preset", "toy",
        "--visual-frames", "2",
        "--visual-h", "2",
        "--visual-w", "2",
        "--steps", "2",
        "--pixel-t", "2",
        "--pixel-h", "8",
        "--pixel-w", "8",
        "--decoder-dim", "32",
        "--decoder-layers", "2",
        "--decoder-heads", "4",
        "--decoder-patch", "2",
        "--decoder-channels", "4",
        "--image", str(image),
        "--visual-tokenizer", "mock",
        "--ref-h", "16",
        "--ref-w", "16",
        "--out", str(out),
        "--seed", "2",
    ])
    assert rc == 0
    assert out.exists()


def test_aw_eval_writes_metrics(tmp_path: Path) -> None:
    torch.manual_seed(0)
    preds = torch.randn(2, 3, 4, 8, 8)
    targets = torch.randn(2, 3, 4, 8, 8)
    pred_path = tmp_path / "preds.pt"
    target_path = tmp_path / "targets.pt"
    out_path = tmp_path / "metrics.json"
    torch.save(preds, pred_path)
    torch.save(targets, target_path)

    rc = eval_main([
        "--predictions", str(pred_path),
        "--targets", str(target_path),
        "--metrics", "mse", "mae", "psnr", "temporal_consistency", "vbench", "fvd",
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("mse", "mae", "psnr", "temporal_consistency", "fvd", "overall"):
        assert key in data


def test_aw_eval_without_targets(tmp_path: Path) -> None:
    """Only metrics that need both prediction and target should be skipped."""
    preds = torch.randn(1, 3, 4, 8, 8)
    pred_path = tmp_path / "preds.pt"
    out_path = tmp_path / "metrics.json"
    torch.save(preds, pred_path)
    rc = eval_main([
        "--predictions", str(pred_path),
        "--metrics", "temporal_consistency", "vbench",
        "--output", str(out_path),
    ])
    assert rc == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "temporal_consistency" in data
    assert "mse" not in data
    assert "overall" in data
