"""Tests for the text-tokenizer abstraction."""

from __future__ import annotations

import pytest
import torch

from another_world.tokenizers.text import (
    HashTextTokenizer,
    TextTokenizer,
    WhitespaceTextTokenizer,
    build_text_tokenizer,
)


def test_hash_tokenizer_deterministic_within_range() -> None:
    tk = HashTextTokenizer(vocab_size_=64)
    a = tk.encode("hello world")
    b = tk.encode("hello world")
    assert torch.equal(a, b)
    assert (a >= 0).all() and (a < 64).all()


def test_hash_tokenizer_handles_empty() -> None:
    tk = HashTextTokenizer(vocab_size_=16)
    ids = tk.encode("")
    assert ids.shape == (1,)


def test_hash_tokenizer_max_len() -> None:
    tk = HashTextTokenizer(vocab_size_=64, max_len=3)
    assert tk.encode("abcdef").shape[0] == 3


def test_hash_tokenizer_decode_is_debug_string() -> None:
    tk = HashTextTokenizer(vocab_size_=32)
    decoded = tk.decode(tk.encode("hi"))
    assert "<h" in decoded


def test_whitespace_tokenizer_round_trip() -> None:
    tk = WhitespaceTextTokenizer(vocab_size_=64)
    text = "another world model"
    ids = tk.encode(text)
    assert ids.dtype == torch.long
    assert tk.decode(ids) == text


def test_whitespace_tokenizer_unk_for_oov_when_full() -> None:
    tk = WhitespaceTextTokenizer(vocab_size_=4)  # only pad/unk + 2 slots
    tk.encode("a b")
    # Third intern attempt should fall back to <unk>.
    ids = tk.encode("c")
    assert tk.decode(ids) == "<unk>"


def test_whitespace_tokenizer_fit_grows_vocab() -> None:
    tk = WhitespaceTextTokenizer(vocab_size_=32)
    tk.fit(["a quick brown fox", "jumps over the lazy dog"])
    ids = tk.encode("fox")
    assert tk.decode(ids) == "fox"


def test_build_text_tokenizer_hash() -> None:
    tk = build_text_tokenizer("hash", vocab_size=16)
    assert isinstance(tk, HashTextTokenizer)
    assert tk.vocab_size == 16


def test_build_text_tokenizer_whitespace() -> None:
    tk = build_text_tokenizer("whitespace", vocab_size=16)
    assert isinstance(tk, WhitespaceTextTokenizer)


def test_build_text_tokenizer_unknown() -> None:
    with pytest.raises(ValueError):
        build_text_tokenizer("bogus")


def test_text_tokenizer_protocol_runtime_check() -> None:
    tk = HashTextTokenizer(vocab_size_=8)
    assert isinstance(tk, TextTokenizer)
