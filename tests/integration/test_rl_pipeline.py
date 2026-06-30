"""Integration: RL trajectories -> SequencePacker -> multimodal trainer."""

from __future__ import annotations

import torch

from another_world.data.datasets import (
    RLTrajectory,
    RLTrajectoryDataset,
    SequencePacker,
)
from another_world.inference.first_frame import MockFirstFrameTokenizer
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.action import DiscreteActionTokenizer
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    run_multimodal_training,
)


def test_rl_trajectories_train_through_pipeline() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()

    trajs = [
        RLTrajectory(
            frames=torch.randn(5, 3, 16, 16),
            actions=torch.randint(0, 4, (5,)),
        )
        for _ in range(8)
    ]

    ds = RLTrajectoryDataset(
        trajectories=trajs,
        visual_tokenizer=MockFirstFrameTokenizer(
            vocab_size=layout.visual_size,
            downsample_spatial=8,
            downsample_temporal=4,
        ),
        action_tokenizer=DiscreteActionTokenizer(vocab_size_=4),
        loops=2,
    )

    packer = SequencePacker(layout, max_len=64)
    samples = list(ds)
    batches = []
    for i in range(0, len(samples), 2):
        chunk = samples[i:i + 2]
        if len(chunk) == 2:
            batches.append(packer.pack_batch(chunk))

    model = MultimodalDynamicsModel(MultimodalDynamicsConfig.toy(layout.total_size))
    history = run_multimodal_training(
        model,
        batches,
        MultimodalTrainerConfig(
            steps=10, lr=3e-3, warmup_steps=2, log_every=5,
            device="cpu", precision="fp32", seed=0,
        ),
    )
    assert len(history) >= 2
    assert history[-1].loss < history[0].loss
