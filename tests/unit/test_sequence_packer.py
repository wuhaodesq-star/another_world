"""Tests for SequencePacker."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.tokenizers.vocab import VocabLayout


def _make_sample(
    text_len: int = 4,
    vt: int = 2, vh: int = 2, vw: int = 2,
    action_len: int | None = None,
) -> TokenSample:
    text = torch.randint(0, 16, (text_len,), dtype=torch.long) if text_len else None
    visual = torch.randint(0, 32, (vt, vh, vw), dtype=torch.long)
    action = (
        torch.randint(0, 8, (action_len,), dtype=torch.long) if action_len else None
    )
    return TokenSample(
        visual_tokens=visual,
        text_tokens=text,
        action_tokens=action,
        key="s0",
        extra={"caption": "x"},
    )


def test_pack_sample_token_count() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=128)
    sample = _make_sample(text_len=4, vt=2, vh=2, vw=2)
    p = packer.pack_sample(sample)
    # 1 BOS + 1 BOC + 4 text + 1 EOC + 1 BOV + 8 visual + 1 EOV + 1 EOS = 18
    assert int(p["tokens"].shape[0]) == 18


def test_pack_sample_modality_codes() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=128)
    sample = _make_sample(text_len=2, vt=1, vh=2, vw=2)
    p = packer.pack_sample(sample)
    expected = [
        3,        # BOS
        3,        # BOC
        0, 0,     # text (2)
        3,        # EOC
        3,        # BOV
        1, 1, 1, 1,  # visual (4)
        3,        # EOV
        3,        # EOS
    ]
    assert p["modality"].tolist() == expected


def test_pack_sample_visual_thw_coords() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=128, include_text=False)
    sample = _make_sample(text_len=0, vt=1, vh=2, vw=2)
    p = packer.pack_sample(sample)
    # Layout (no text): BOS, BOV, v(0,0), v(0,1), v(1,0), v(1,1), EOV, EOS
    h_coords = p["h"].tolist()
    w_coords = p["w"].tolist()
    # positions 2..5 are the visual tokens
    assert h_coords[2:6] == [0, 0, 1, 1]
    assert w_coords[2:6] == [0, 1, 0, 1]
    # Non-visual rows have zeros.
    assert h_coords[0] == 0 and h_coords[1] == 0
    assert h_coords[-1] == 0


def test_pack_sample_token_ids_in_proper_slabs() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=128)
    sample = _make_sample(text_len=3, vt=1, vh=2, vw=2)
    p = packer.pack_sample(sample)
    # Positions: 0=BOS, 1=BOC, 2..4=text(3), 5=EOC, 6=BOV, 7..10=visual(4),
    # 11=EOV, 12=EOS
    text_ids = p["tokens"][2:5]
    visual_ids = p["tokens"][7:11]
    assert ((text_ids >= 0) & (text_ids < layout.visual_start)).all()
    assert (
        (visual_ids >= layout.visual_start)
        & (visual_ids < layout.action_start)
    ).all()


def test_pack_batch_pads_and_builds_axes() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    samples = [_make_sample(text_len=2, vt=1, vh=2, vw=2),
               _make_sample(text_len=4, vt=2, vh=2, vw=2)]
    batch = packer.pack_batch(samples)
    assert batch.tokens.shape == (2, 24)
    assert batch.axes.modality.shape == (2, 24)
    assert batch.targets.shape == (2, 24)
    assert batch.loss_mask.shape == (2, 24)
    assert batch.lengths.tolist() == [
        # sample 0: bos+boc+2 text+eoc+bov+4 visual+eov+eos = 12
        12,
        # sample 1: bos+boc+4 text+eoc+bov+8 visual+eov+eos = 18
        18,
    ]
    # Beyond the length, tokens must be PAD and targets must be -100.
    for i, n in enumerate(batch.lengths.tolist()):
        assert (batch.tokens[i, n:] == packer.vocab.pad_id).all()
        assert (batch.targets[i, max(n - 1, 0):] == -100).all()
        assert (batch.loss_mask[i, max(n - 1, 0):] == 0.0).all()


def test_pack_batch_score_only_visual_by_default() -> None:
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32)
    samples = [_make_sample(text_len=2, vt=1, vh=2, vw=2)]
    batch = packer.pack_batch(samples)
    # loss_mask is shifted: position i scores prediction of token i+1.
    # So the score positions are those *before* a visual token in `tokens`.
    target_modality = batch.axes.modality[:, 1:]
    expected_score = (target_modality == 1).to(torch.float32)
    # pad pre-pended after shift
    expected = torch.zeros_like(batch.loss_mask)
    expected[:, :-1] = expected_score
    expected[:, max(int(batch.lengths[0].item()) - 1, 0):] = 0
    assert torch.allclose(batch.loss_mask, expected)
