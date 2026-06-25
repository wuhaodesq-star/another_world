"""Integration: trainer writes a checkpoint, then resumes from it."""

from __future__ import annotations

from pathlib import Path

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.checkpoint import find_latest_checkpoint, load_checkpoint
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    run_multimodal_training,
)


def _batches(layout, packer, n_batches, batch_size=2):
    out = []
    for i in range(n_batches):
        chunk = [
            TokenSample(
                visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
                text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
                key=f"k{i}-{j}",
            )
            for j in range(batch_size)
        ]
        out.append(packer.pack_batch(chunk))
    return out


def test_periodic_save_and_resume(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)

    ckpt_root = tmp_path / "ckpts"
    cfg = MultimodalTrainerConfig(
        steps=10, lr=1e-3, warmup_steps=1, log_every=5,
        device="cpu", precision="fp32", seed=0,
        checkpoint_dir=str(ckpt_root),
        checkpoint_every=5,
        checkpoint_keep=2,
    )
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    run_multimodal_training(model, _batches(layout, packer, 20), cfg)

    # We expect at least one saved snapshot, kept under the keep cap.
    snapshots = sorted(ckpt_root.iterdir())
    assert len(snapshots) >= 1
    assert len(snapshots) <= cfg.checkpoint_keep

    latest = find_latest_checkpoint(ckpt_root)
    assert latest is not None
    assert latest.name.startswith("step-")

    # Resume into a fresh model, verify parameters match.
    fresh = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    # Fresh init differs from trained model first.
    assert not torch.equal(
        fresh.tok_embeddings.weight, model.tok_embeddings.weight
    )
    meta = load_checkpoint(latest, model=fresh)
    assert meta.step > 0
    assert torch.equal(
        fresh.tok_embeddings.weight, model.tok_embeddings.weight
    )
