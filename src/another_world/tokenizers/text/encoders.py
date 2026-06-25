"""Text tokenizer wrappers.

Three implementations sharing :class:`TextTokenizer` protocol:

- :class:`HashTextTokenizer`        : deterministic hash-based mapping used by
                                     tests and laptop smoke runs (no network).
- :class:`WhitespaceTextTokenizer`  : simple vocabulary built from observed
                                     tokens, useful for small offline corpora.
- :class:`HFTextTokenizer`          : thin wrapper around any HuggingFace
                                     tokenizer (LLaMA-3 BPE by default).

The factory :func:`build_text_tokenizer` picks one by name.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import torch
from torch import Tensor

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


@runtime_checkable
class TextTokenizer(Protocol):
    """A small interface every text tokenizer must satisfy."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> Tensor: ...

    def decode(self, ids: Tensor | list[int]) -> str: ...


# ---------------------------------------------------------------------------
# Hash tokenizer (offline)
# ---------------------------------------------------------------------------


@dataclass
class HashTextTokenizer:
    """Deterministic character-level hash tokenizer.

    Used by unit tests and offline demos when no real tokenizer is
    available. ``encode(s)`` maps each character to an id in
    ``[0, vocab_size)`` via SHA-1.
    """

    vocab_size_: int = 256
    max_len: int | None = 256

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_

    def encode(self, text: str) -> Tensor:
        ids: list[int] = []
        for i, ch in enumerate(text):
            digest = hashlib.sha1(f"{i}-{ch}".encode("utf-8")).digest()
            ids.append(int.from_bytes(digest[:2], "big") % self.vocab_size_)
            if self.max_len is not None and len(ids) >= self.max_len:
                break
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Tensor | list[int]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        # Hash is non-invertible; return a debug stringification.
        return " ".join(f"<h{int(i):x}>" for i in ids)


# ---------------------------------------------------------------------------
# Whitespace tokenizer
# ---------------------------------------------------------------------------


@dataclass
class WhitespaceTextTokenizer:
    """Word-level tokenizer with an in-memory growing vocabulary.

    Useful for tests that need an *invertible* tokenizer without pulling
    in transformers / tokenizers.
    """

    vocab_size_: int = 1024
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    _vocab: dict[str, int] = field(default_factory=dict, init=False)
    _inverse: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Reserve slot 0 for <pad> and slot 1 for <unk> before allowing any
        # other tokens to be interned.
        self._vocab[self.pad_token] = 0
        self._inverse.append(self.pad_token)
        self._vocab[self.unk_token] = 1
        self._inverse.append(self.unk_token)

    def _intern(self, token: str) -> int:
        if token in self._vocab:
            return self._vocab[token]
        if len(self._vocab) < self.vocab_size_:
            self._vocab[token] = len(self._vocab)
            self._inverse.append(token)
            return self._vocab[token]
        return self._vocab[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_

    def fit(self, corpus: Iterable[str]) -> "WhitespaceTextTokenizer":
        for text in corpus:
            for tok in text.split():
                self._intern(tok)
        return self

    def encode(self, text: str) -> Tensor:
        ids = [self._intern(t) for t in text.split()]
        if not ids:
            ids = [self._vocab[self.pad_token]]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Tensor | list[int]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        out: list[str] = []
        for i in ids:
            i = int(i)
            if i < len(self._inverse):
                out.append(self._inverse[i])
            else:
                out.append(self.unk_token)
        return " ".join(out)


# ---------------------------------------------------------------------------
# HuggingFace wrapper
# ---------------------------------------------------------------------------


@dataclass
class HFTextTokenizer:
    """Thin wrapper around any ``transformers.PreTrainedTokenizer``."""

    model_name: str = "meta-llama/Meta-Llama-3-8B"
    cache_dir: str | None = None
    max_len: int | None = 2048

    _backend: object = field(default=None, init=False, repr=False)

    def _ensure_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for HFTextTokenizer "
                "(`pip install transformers`)."
            ) from exc
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]
        self._backend = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        return self._backend

    @property
    def vocab_size(self) -> int:
        tk = self._ensure_backend()
        return int(tk.vocab_size)

    def encode(self, text: str) -> Tensor:
        tk = self._ensure_backend()
        out = tk(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
        )
        return out["input_ids"][0].to(torch.long)

    def decode(self, ids: Tensor | list[int]) -> str:
        tk = self._ensure_backend()
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return tk.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_text_tokenizer(
    kind: str = "hash",
    *,
    vocab_size: int = 256,
    hf_model: str = "meta-llama/Meta-Llama-3-8B",
    max_len: int | None = 256,
) -> TextTokenizer:
    """Instantiate a tokenizer by name."""

    if kind == "hash":
        return HashTextTokenizer(vocab_size_=vocab_size, max_len=max_len)
    if kind == "whitespace":
        return WhitespaceTextTokenizer(vocab_size_=vocab_size)
    if kind == "hf":
        return HFTextTokenizer(model_name=hf_model, max_len=max_len)
    raise ValueError(
        f"unknown text tokenizer kind '{kind}' "
        f"(expected one of: hash, whitespace, hf)"
    )


__all__ = [
    "HFTextTokenizer",
    "HashTextTokenizer",
    "TextTokenizer",
    "WhitespaceTextTokenizer",
    "build_text_tokenizer",
]
