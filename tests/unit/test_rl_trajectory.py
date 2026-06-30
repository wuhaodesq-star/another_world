"""Tests for the RL trajectory dataset."""

from __future__ import annotations

import pytest
import torch

from another_world.data.datasets.rl_trajectory import (
    RLTrajectory,
    RLTrajectoryDataset,
    trajectories_from_dicts,
)
from another_world.inference.first_frame import MockFirstFrameTokenizer
from another_world.tokenizers.action import (
    BinnedActionTokenizer,
    DiscreteActionTokenizer,
)


def _traj_discrete(T: int = 5) -> RLTrajectory:
    return RLTrajectory(
        frames=torch.randn(T, 3, 16, 16),
        actions=torch.randint(0, 6, (T,)),
        caption="discrete episode",
    )


def _traj_continuous(T: int = 5, D: int = 4) -> RLTrajectory:
    return RLTrajectory(
        frames=torch.randn(T, 3, 16, 16),
        actions=torch.randn(T, D).clamp(-1, 1),
    )


def test_rl_trajectory_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        RLTrajectory(
            frames=torch.zeros(5, 3, 8, 8),
            actions=torch.zeros(4),
        )


def test_rl_trajectory_rejects_bad_frame_rank() -> None:
    with pytest.raises(ValueError):
        RLTrajectory(frames=torch.zeros(5, 3, 8), actions=torch.zeros(5))


def test_rl_trajectory_rejects_bad_action_rank() -> None:
    with pytest.raises(ValueError):
        RLTrajectory(
            frames=torch.zeros(5, 3, 8, 8),
            actions=torch.zeros(5, 4, 2),
        )


def test_rl_dataset_yields_token_samples_discrete() -> None:
    trajs = [_traj_discrete() for _ in range(3)]
    ds = RLTrajectoryDataset(
        trajectories=trajs,
        visual_tokenizer=MockFirstFrameTokenizer(vocab_size=128),
        action_tokenizer=DiscreteActionTokenizer(vocab_size_=6),
    )
    samples = list(ds)
    assert len(samples) == 3
    s = samples[0]
    assert s.visual_tokens.dtype == torch.long
    assert s.action_tokens is not None
    assert s.action_tokens.shape[0] == 5
    assert s.extra["caption"] == "discrete episode"


def test_rl_dataset_yields_token_samples_continuous() -> None:
    trajs = [_traj_continuous() for _ in range(2)]
    ds = RLTrajectoryDataset(
        trajectories=trajs,
        visual_tokenizer=MockFirstFrameTokenizer(vocab_size=64),
        action_tokenizer=BinnedActionTokenizer(dim=4, bins=8, low=-1.0, high=1.0),
    )
    samples = list(ds)
    assert len(samples) == 2
    s = samples[0]
    assert s.visual_tokens.dtype == torch.long
    # binned tokenizer: 4 channels per step * 5 steps = 20 tokens
    assert s.action_tokens.shape == (20,)


def test_rl_dataset_loops_factor() -> None:
    trajs = [_traj_discrete() for _ in range(2)]
    ds = RLTrajectoryDataset(
        trajectories=trajs,
        visual_tokenizer=MockFirstFrameTokenizer(vocab_size=32),
        action_tokenizer=None,
        loops=3,
    )
    samples = list(ds)
    assert len(samples) == 6


def test_rl_dataset_skips_failing_trajectory() -> None:
    class _Bomb:
        def encode(self, video):
            raise RuntimeError("boom")

    trajs = [_traj_discrete()]
    ds = RLTrajectoryDataset(trajectories=trajs, visual_tokenizer=_Bomb())
    # Should not raise; failing trajectories are skipped.
    samples = list(ds)
    assert samples == []


def test_trajectories_from_dicts() -> None:
    raw = [
        {
            "frames": torch.zeros(3, 3, 8, 8),
            "actions": torch.zeros(3),
            "caption": "hi",
            "fps": 24,
        },
    ]
    out = trajectories_from_dicts(raw)
    assert len(out) == 1
    assert out[0].caption == "hi"
    assert out[0].extra == {"fps": 24}
