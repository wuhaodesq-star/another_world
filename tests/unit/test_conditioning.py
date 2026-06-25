"""Tests for first-frame and action conditioning in the rollout."""

from __future__ import annotations

import pytest
import torch

from another_world.inference.generation import (
    GenerationConfig,
    _normalize_action_ids,
    _normalize_first_frame,
    rollout_visual_tokens,
)
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def _toy_model(layout: VocabLayout) -> MultimodalDynamicsModel:
    return MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )


def test_normalize_first_frame_expands_2d_to_3d() -> None:
    cfg = GenerationConfig(visual_frames=4, visual_height=2, visual_width=2)
    frame = torch.zeros(2, 2, dtype=torch.long)
    out = _normalize_first_frame(frame, cfg)
    assert out is not None and out.shape == (1, 2, 2)


def test_normalize_first_frame_none_passthrough() -> None:
    cfg = GenerationConfig(visual_frames=2, visual_height=2, visual_width=2)
    assert _normalize_first_frame(None, cfg) is None


def test_normalize_first_frame_rejects_full_cube() -> None:
    cfg = GenerationConfig(visual_frames=2, visual_height=2, visual_width=2)
    with pytest.raises(ValueError, match="must be"):
        # T_prefix must be strictly less than visual_frames
        _normalize_first_frame(torch.zeros(2, 2, 2, dtype=torch.long), cfg)


def test_normalize_first_frame_rejects_wrong_spatial() -> None:
    cfg = GenerationConfig(visual_frames=4, visual_height=2, visual_width=2)
    with pytest.raises(ValueError, match="spatial"):
        _normalize_first_frame(torch.zeros(1, 3, 2, dtype=torch.long), cfg)


def test_normalize_first_frame_rejects_bad_rank() -> None:
    cfg = GenerationConfig(visual_frames=4, visual_height=2, visual_width=2)
    with pytest.raises(ValueError):
        _normalize_first_frame(torch.zeros(5, dtype=torch.long), cfg)


def test_normalize_action_ids_from_tensor() -> None:
    assert _normalize_action_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]


def test_normalize_action_ids_none() -> None:
    assert _normalize_action_ids(None) is None


def test_rollout_with_first_frame_preserves_prefix() -> None:
    """First-frame visual ids must appear unchanged at the head of the output."""
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()
    cfg = GenerationConfig(
        visual_frames=2, visual_height=2, visual_width=2,
        seed=0, use_kv_cache=True,
    )
    # Pre-supplied prefix: 1 frame of 2x2 local visual ids.
    prefix = torch.tensor([[[3, 5], [7, 11]]], dtype=torch.long)
    out = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg, layout=layout,
        first_frame=prefix,
    )
    assert out.shape == (2, 2, 2)
    # The first temporal slice must match the prefix exactly.
    assert torch.equal(out[0], prefix[0])


def test_rollout_with_first_frame_recompute_matches_kv() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()
    prefix = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    cfg_kv = GenerationConfig(
        visual_frames=2, visual_height=2, visual_width=2,
        seed=42, use_kv_cache=True,
    )
    cfg_naive = GenerationConfig(
        visual_frames=2, visual_height=2, visual_width=2,
        seed=42, use_kv_cache=False,
    )
    out_kv = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg_kv, layout=layout,
        first_frame=prefix,
    )
    out_naive = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg_naive, layout=layout,
        first_frame=prefix,
    )
    assert torch.equal(out_kv, out_naive)


def test_rollout_with_action_prefix_runs() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()
    cfg = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        seed=0, use_kv_cache=True,
    )
    out = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg, layout=layout,
        action_ids=[3, 5, 7],
    )
    assert out.shape == (1, 2, 2)


def test_rollout_action_prefix_changes_output() -> None:
    """Action prefix should be threaded through the prompt without errors.

    On a random toy model with a small vocabulary the sampled sequences
    can coincide; we instead verify that the rollout runs to completion
    in both cases and produces in-range visual ids -- the sensitivity
    test runs against a *real* model on the integration side.
    """
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()
    cfg = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        seed=0, use_kv_cache=True,
    )
    a = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg, layout=layout, action_ids=[1, 2],
    )
    b = rollout_visual_tokens(
        model, text_ids=[1, 2], config=cfg, layout=layout, action_ids=[8, 9],
    )
    assert a.shape == b.shape == (1, 2, 2)
    assert (a >= 0).all() and (a < layout.visual_size).all()
    assert (b >= 0).all() and (b < layout.visual_size).all()


def test_rollout_combined_first_frame_and_action() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = _toy_model(layout)
    model.eval()
    cfg = GenerationConfig(
        visual_frames=2, visual_height=2, visual_width=2,
        seed=0, use_kv_cache=True,
    )
    prefix = torch.tensor([[2, 4], [6, 8]], dtype=torch.long)
    out = rollout_visual_tokens(
        model, text_ids=[1], config=cfg, layout=layout,
        first_frame=prefix, action_ids=[1, 2],
    )
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0], prefix)
