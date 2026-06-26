"""Tests for the CLIPTextEncoder wrapper (mocking transformers)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from another_world.tokenizers.text import CLIPTextEncoder


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_name: str):
        inst = cls()
        inst.model_name = model_name
        return inst

    def __call__(self, texts, **kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


class _FakeTextModel:
    @classmethod
    def from_pretrained(cls, model_name: str):
        inst = cls()
        inst.model_name = model_name
        return inst

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        b = inputs["input_ids"].shape[0]
        # deterministic embeddings
        emb = torch.arange(b * 8, dtype=torch.float32).view(b, 8)
        return SimpleNamespace(pooler_output=emb)


def test_clip_text_encoder_encode_mocked_transformers() -> None:
    fake_transformers = MagicMock(
        CLIPTokenizer=_FakeTokenizer,
        CLIPTextModel=_FakeTextModel,
    )
    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        enc = CLIPTextEncoder(model_name="fake/clip", device="cpu", normalize=True)
        out = enc.encode(["hello", "world"])
    assert out.shape == (2, 8)
    # normalized rows
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_clip_text_encoder_similarity_shape() -> None:
    fake_transformers = MagicMock(
        CLIPTokenizer=_FakeTokenizer,
        CLIPTextModel=_FakeTextModel,
    )
    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        enc = CLIPTextEncoder(device="cpu")
        sim = enc.similarity(["a", "b"], ["c", "d", "e"])
    assert sim.shape == (2, 3)


def test_clip_text_encoder_single_string() -> None:
    fake_transformers = MagicMock(
        CLIPTokenizer=_FakeTokenizer,
        CLIPTextModel=_FakeTextModel,
    )
    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        enc = CLIPTextEncoder(device="cpu", normalize=False)
        out = enc.encode("hello")
    assert out.shape == (1, 8)


def test_clip_text_encoder_missing_transformers_raises() -> None:
    # Ensure a missing module produces a clean ImportError.
    with patch.dict("sys.modules", {"transformers": None}):
        enc = CLIPTextEncoder(device="cpu")
        with pytest.raises(ImportError):
            enc.encode("hello")
