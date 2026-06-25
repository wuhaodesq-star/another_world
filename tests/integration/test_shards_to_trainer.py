"""Integration test: token shards -> packed batches -> multimodal trainer."""

from __future__ import annotations

from pathlib import Path

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.data.datasets.token_shard_stream import (
    build_packed_batch_stream,
)
from another_world.data.tokenize import TokenShardWriter
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    run_multimodal_training,
)


def test_train_from_shards_loss_decreases(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    # Write 8 token samples to a single shard.
    shard = tmp_path / "shard-00000.tar"
    with TokenShardWriter(path=shard) as w:
        for i in range(8):
            w.append(
                TokenSample(
                    visual_tokens=torch.randint(
                        0, 16, (1, 2, 2), dtype=torch.long
                    ),
                    text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
                    key=f"k{i}",
                )
            )

    packer = SequencePacker(layout, max_len=32)
    batches = build_packed_batch_stream(
        [str(shard)],
        packer=packer,
        batch_size=2,
        loops=10**4,  # cycle forever
        drop_last=True,
    )
    cfg = MultimodalDynamicsConfig.toy(layout.total_size)
    model = MultimodalDynamicsModel(cfg)

    history = run_multimodal_training(
        model, batches,
        MultimodalTrainerConfig(
            steps=20, lr=5e-3, warmup_steps=2, log_every=5,
            device="cpu", precision="fp32", seed=0,
        ),
    )
    assert history[-1].loss < history[0].loss
