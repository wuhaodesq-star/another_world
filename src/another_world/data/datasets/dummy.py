"""Dummy in-memory datasets used for unit tests and smoke training runs."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A deterministic dataset of random token sequences.

    Each item yields ``(input_ids, target_ids)`` where ``target_ids`` are the
    inputs shifted by one position, matching the standard next-token training
    setup.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        seq_len: int = 128,
        length: int = 256,
        seed: int = 0,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if length < 1:
            raise ValueError("length must be >= 1")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

        generator = torch.Generator().manual_seed(seed)
        # Pre-generate so the dataset is deterministic and cheap to index.
        self._tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(length, seq_len + 1),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < self.length:
            raise IndexError(idx)
        row = self._tokens[idx]
        inputs = row[:-1].clone()
        targets = row[1:].clone()
        return inputs, targets


__all__ = ["DummyTokenDataset"]
