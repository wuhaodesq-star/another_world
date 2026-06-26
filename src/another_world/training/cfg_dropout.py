"""Training-time conditioning dropout for classifier-free guidance.

To support CFG at inference we need the *training* signal to include
unconditional examples. The standard recipe (Ho & Salimans, 2022) drops
each conditioning channel independently with some probability, replacing
its tokens with neutral / null fillers and zeroing their loss
contribution.

For our packed multimodal sequences we drop the text channel and / or
the action channel by replacing those tokens with a null token id. This
keeps the sequence shape intact (axes / linear positions remain
identical) which is what the trainer's downstream code expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from another_world.data.datasets.sequence_packer import PackedBatch
from another_world.tokenizers.vocab import VocabInfo


@dataclass
class ConditioningDropout:
    """Per-batch conditioning-mask sampler.

    Args:
        vocab: optional :class:`VocabInfo`; when present, dropped tokens are
            replaced by ``vocab.unk_id``.
        null_token_id: explicit replacement id. Takes precedence over
            ``vocab.unk_id``. This is useful when the trainer does not know
            the exact vocab layout (e.g. tiny vs default) but the caller can
            pass a safe null id.
        text_drop_prob: probability per *sample* of dropping the text channel.
        action_drop_prob: probability per *sample* of dropping the action channel.
        seed: optional RNG seed for reproducible dropout.
    """

    vocab: VocabInfo | None = None
    null_token_id: int | None = None
    text_drop_prob: float = 0.1
    action_drop_prob: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        for name, p in (
            ("text_drop_prob", self.text_drop_prob),
            ("action_drop_prob", self.action_drop_prob),
        ):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {p}")
        if self.vocab is None and self.null_token_id is None:
            raise ValueError("provide either vocab or null_token_id")
        self._generator = (
            torch.Generator().manual_seed(self.seed)
            if self.seed is not None else None
        )

    @property
    def replacement_id(self) -> int:
        if self.null_token_id is not None:
            return int(self.null_token_id)
        assert self.vocab is not None
        return int(self.vocab.unk_id)

    def __call__(self, batch: PackedBatch) -> PackedBatch:
        """Return a new :class:`PackedBatch` with conditioning channels dropped."""

        if self.text_drop_prob == 0.0 and self.action_drop_prob == 0.0:
            return batch

        bsz = batch.tokens.shape[0]
        null_id = self.replacement_id
        device = batch.tokens.device

        rand = (
            torch.rand(bsz, 2, generator=self._generator).to(device)
            if self._generator is not None else torch.rand(bsz, 2, device=device)
        )
        drop_text = rand[:, 0] < self.text_drop_prob
        drop_action = rand[:, 1] < self.action_drop_prob

        tokens = batch.tokens.clone()
        modality = batch.axes.modality

        if drop_text.any():
            mask = drop_text[:, None] & (modality == 0)
            tokens = torch.where(mask, torch.full_like(tokens, null_id), tokens)
        if drop_action.any():
            mask = drop_action[:, None] & (modality == 2)
            tokens = torch.where(mask, torch.full_like(tokens, null_id), tokens)

        new_targets = torch.full_like(tokens, fill_value=-100)
        new_targets[:, :-1] = tokens[:, 1:]
        new_targets = torch.where(batch.targets == -100, batch.targets, new_targets)

        return PackedBatch(
            tokens=tokens,
            targets=new_targets,
            axes=batch.axes,
            loss_mask=batch.loss_mask,
            lengths=batch.lengths,
        )


__all__ = ["ConditioningDropout"]
