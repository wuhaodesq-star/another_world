"""Tests for training-time CFG conditioning dropout."""

from __future__ import annotations

import pytest
import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.tokenizers.vocab import VocabLayout, VocabInfo
from another_world.training.cfg_dropout import ConditioningDropout


def _batch_with_action() -> tuple[VocabLayout, SequencePacker, object]:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32, score_text=True, score_action=True)
    sample = TokenSample(
        visual_tokens=torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long),
        text_tokens=torch.tensor([5, 6, 7], dtype=torch.long),
        action_tokens=torch.tensor([1, 2], dtype=torch.long),
    )
    return layout, packer, packer.pack_batch([sample])


def test_cfg_dropout_noop_returns_same_object() -> None:
    layout, _, batch = _batch_with_action()
    drop = ConditioningDropout(
        vocab=VocabInfo(layout), text_drop_prob=0.0, action_drop_prob=0.0,
    )
    assert drop(batch) is batch


def test_cfg_dropout_replaces_text_tokens() -> None:
    layout, _, batch = _batch_with_action()
    null_id = VocabInfo(layout).unk_id
    drop = ConditioningDropout(
        vocab=VocabInfo(layout), text_drop_prob=1.0, action_drop_prob=0.0,
        seed=0,
    )
    out = drop(batch)
    text_mask = batch.axes.modality == 0
    assert (out.tokens[text_mask] == null_id).all()
    action_mask = batch.axes.modality == 2
    assert torch.equal(out.tokens[action_mask], batch.tokens[action_mask])


def test_cfg_dropout_replaces_action_tokens_with_explicit_null() -> None:
    _, _, batch = _batch_with_action()
    drop = ConditioningDropout(
        null_token_id=123, text_drop_prob=0.0, action_drop_prob=1.0, seed=0,
    )
    out = drop(batch)
    action_mask = batch.axes.modality == 2
    assert (out.tokens[action_mask] == 123).all()
    text_mask = batch.axes.modality == 0
    assert torch.equal(out.tokens[text_mask], batch.tokens[text_mask])


def test_cfg_dropout_recomputes_targets() -> None:
    _, _, batch = _batch_with_action()
    drop = ConditioningDropout(
        null_token_id=0, text_drop_prob=1.0, action_drop_prob=1.0, seed=0,
    )
    out = drop(batch)
    valid = batch.targets[:, :-1] != -100
    assert torch.equal(out.targets[:, :-1][valid], out.tokens[:, 1:][valid])
    # Preserve ignored tail positions.
    assert (out.targets[batch.targets == -100] == -100).all()


def test_cfg_dropout_requires_null_or_vocab() -> None:
    with pytest.raises(ValueError):
        ConditioningDropout(text_drop_prob=0.1)


def test_cfg_dropout_probability_validation() -> None:
    with pytest.raises(ValueError):
        ConditioningDropout(null_token_id=0, text_drop_prob=-0.1)
    with pytest.raises(ValueError):
        ConditioningDropout(null_token_id=0, action_drop_prob=1.1)
