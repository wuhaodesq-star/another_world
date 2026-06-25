"""Tests for the joint multimodal vocabulary layout."""

from __future__ import annotations

import pytest

from another_world.tokenizers.vocab import VocabInfo, VocabLayout


def test_default_layout_sizes() -> None:
    layout = VocabLayout.default()
    assert layout.total_size == 32_000 + 65_536 + 4_096 + 256
    assert layout.text_start == 0
    assert layout.visual_start == 32_000
    assert layout.action_start == 32_000 + 65_536


def test_tiny_layout_is_small() -> None:
    layout = VocabLayout.tiny()
    assert layout.total_size == 64 + 128 + 16 + 16


def test_encode_round_trip_text() -> None:
    layout = VocabLayout.tiny()
    gid = layout.encode_text(5)
    assert layout.modality_of(gid) == "text"
    assert gid == 5


def test_encode_round_trip_visual() -> None:
    layout = VocabLayout.tiny()
    gid = layout.encode_visual(7)
    assert layout.modality_of(gid) == "visual"
    assert gid == layout.visual_start + 7


def test_encode_round_trip_action() -> None:
    layout = VocabLayout.tiny()
    gid = layout.encode_action(3)
    assert layout.modality_of(gid) == "action"
    assert gid == layout.action_start + 3


def test_special_ids_lookup() -> None:
    layout = VocabLayout.tiny()
    pad = layout.special_id("pad")
    assert layout.modality_of(pad) == "special"
    assert pad == layout.special_start


def test_unknown_special_raises() -> None:
    layout = VocabLayout.tiny()
    with pytest.raises(KeyError):
        layout.special_id("nope")


def test_out_of_range_encode_raises() -> None:
    layout = VocabLayout.tiny()
    with pytest.raises(ValueError):
        layout.encode_text(layout.text_size)
    with pytest.raises(ValueError):
        layout.encode_visual(-1)
    with pytest.raises(ValueError):
        layout.encode_action(layout.action_size + 5)


def test_modality_of_rejects_oob() -> None:
    layout = VocabLayout.tiny()
    with pytest.raises(ValueError):
        layout.modality_of(layout.total_size + 1)


def test_vocab_info_populates_all_ids() -> None:
    info = VocabInfo(VocabLayout.tiny())
    for name in (
        "pad", "bos", "eos", "bov", "eov", "boc", "eoc", "boa", "eoa", "mask",
    ):
        assert hasattr(info, f"{name}_id")
    assert info.pad_id == info.layout.special_id("pad")
    assert info.bos_id == info.layout.special_id("bos")
