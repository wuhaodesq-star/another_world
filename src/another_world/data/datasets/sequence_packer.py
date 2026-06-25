"""Pack heterogenous multimodal samples into fixed-shape training batches.

Each :class:`~another_world.data.datasets.sample.TokenSample` carries
visual tokens (typically a 3-D ``[T, H, W]`` index volume) and optional
text tokens.  The dynamics model wants a single flat sequence per sample
with parallel :class:`RopeAxes` arrays so the attention kernel knows
which slot in each token's head dim corresponds to which axis.

:class:`SequencePacker` performs the packing:

1. flattens visual tokens row-major into ``T*H*W`` ids,
2. surrounds them with ``bov`` / ``eov`` (and similarly for text),
3. shifts every modality's local ids into the *global* slabs defined by
   :class:`~another_world.tokenizers.vocab.VocabLayout`,
4. builds matching modality / linear / t / h / w arrays,
5. left-truncates or right-pads to a fixed ``max_len``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from another_world.data.datasets.sample import TokenSample
from another_world.models.layers.mixed_rope import RopeAxes
from another_world.tokenizers.vocab import VocabInfo, VocabLayout


@dataclass
class PackedBatch:
    """A model-ready batch."""

    tokens: Tensor       # [B, T] long
    targets: Tensor      # [B, T] long; -100 == ignore
    axes: RopeAxes       # each field [B, T] long
    loss_mask: Tensor    # [B, T] float; rows we score get 1
    lengths: Tensor      # [B] long; actual sequence length before padding

    def to(self, device: torch.device) -> "PackedBatch":
        return PackedBatch(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            axes=self.axes.to(device),
            loss_mask=self.loss_mask.to(device),
            lengths=self.lengths.to(device),
        )


@dataclass
class SequencePacker:
    """Build :class:`PackedBatch` objects from raw :class:`TokenSample` s.

    Args:
        layout: the joint vocabulary layout.
        max_len: padding length (longer samples are right-truncated).
        include_text: whether to include the text segment in the sequence.
        score_text / score_visual: which modalities contribute to the loss
            mask. Defaults are tuned for visual-conditioned training: we
            train the model to predict visual tokens but treat text as
            already-known context.
    """

    layout: VocabLayout
    max_len: int = 1024
    include_text: bool = True
    score_text: bool = False
    score_visual: bool = True
    score_action: bool = True

    vocab: VocabInfo = field(init=False)

    def __post_init__(self) -> None:
        self.vocab = VocabInfo(layout=self.layout)

    # ----- per-sample packing --------------------------------------------

    def pack_sample(self, sample: TokenSample) -> dict[str, Tensor]:
        """Return per-field 1-D tensors of length ``L`` (no padding yet)."""

        tokens: list[int] = []
        modality: list[int] = []
        linear: list[int] = []
        t_coords: list[int] = []
        h_coords: list[int] = []
        w_coords: list[int] = []
        score: list[int] = []

        # ----- BOS ----------------------------------------------------
        tokens.append(self.vocab.bos_id)
        modality.append(3)
        score.append(0)

        # ----- text segment ------------------------------------------
        if self.include_text and sample.text_tokens is not None:
            tokens.append(self.vocab.boc_id)
            modality.append(3)
            score.append(1 if self.score_text else 0)
            for tid in sample.text_tokens.reshape(-1).tolist():
                tokens.append(self.layout.encode_text(int(tid)))
                modality.append(0)
                score.append(1 if self.score_text else 0)
            tokens.append(self.vocab.eoc_id)
            modality.append(3)
            score.append(0)

        # ----- visual segment ----------------------------------------
        visual = sample.visual_tokens
        if visual is None:
            raise ValueError("sample has no visual_tokens")
        if visual.dim() == 4 and visual.shape[0] == 1:
            visual = visual[0]
        if visual.dim() != 3:
            raise ValueError(
                f"expected visual_tokens [T, H, W] (got {tuple(visual.shape)})"
            )
        t_dim, h_dim, w_dim = visual.shape
        tokens.append(self.vocab.bov_id)
        modality.append(3)
        score.append(0)
        visual_flat = visual.reshape(-1).tolist()
        idx = 0
        for ti in range(t_dim):
            for hi in range(h_dim):
                for wi in range(w_dim):
                    tokens.append(self.layout.encode_visual(int(visual_flat[idx])))
                    modality.append(1)
                    score.append(1 if self.score_visual else 0)
                    t_coords.append(ti)
                    h_coords.append(hi)
                    w_coords.append(wi)
                    idx += 1
        tokens.append(self.vocab.eov_id)
        modality.append(3)
        score.append(0)

        # ----- action segment ----------------------------------------
        if sample.action_tokens is not None:
            tokens.append(self.vocab.boa_id)
            modality.append(3)
            score.append(0)
            for aid in sample.action_tokens.reshape(-1).tolist():
                tokens.append(self.layout.encode_action(int(aid)))
                modality.append(2)
                score.append(1 if self.score_action else 0)
            tokens.append(self.vocab.eoa_id)
            modality.append(3)
            score.append(0)

        # ----- EOS ----------------------------------------------------
        tokens.append(self.vocab.eos_id)
        modality.append(3)
        score.append(0)

        # ----- per-position metadata ---------------------------------
        n = len(tokens)
        # Linear positions are dense within this sample.
        linear = list(range(n))
        # Pad t/h/w to full length with zeros for non-visual rows.
        full_t = [0] * n
        full_h = [0] * n
        full_w = [0] * n
        v_idx = 0
        for i, m in enumerate(modality):
            if m == 1:
                full_t[i] = t_coords[v_idx]
                full_h[i] = h_coords[v_idx]
                full_w[i] = w_coords[v_idx]
                v_idx += 1

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "modality": torch.tensor(modality, dtype=torch.long),
            "linear": torch.tensor(linear, dtype=torch.long),
            "t": torch.tensor(full_t, dtype=torch.long),
            "h": torch.tensor(full_h, dtype=torch.long),
            "w": torch.tensor(full_w, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.long),
        }

    # ----- batch packing --------------------------------------------------

    def pack_batch(self, samples: list[TokenSample]) -> PackedBatch:
        """Pack a list of samples into a single :class:`PackedBatch`."""

        per_sample = [self.pack_sample(s) for s in samples]
        bsz = len(per_sample)
        # Trim or pad each sample to ``max_len``.
        T = self.max_len

        tokens = torch.full((bsz, T), self.vocab.pad_id, dtype=torch.long)
        modality = torch.full((bsz, T), 3, dtype=torch.long)  # special
        linear = torch.zeros((bsz, T), dtype=torch.long)
        t = torch.zeros((bsz, T), dtype=torch.long)
        h = torch.zeros((bsz, T), dtype=torch.long)
        w = torch.zeros((bsz, T), dtype=torch.long)
        score = torch.zeros((bsz, T), dtype=torch.long)
        lengths = torch.zeros((bsz,), dtype=torch.long)

        for i, p in enumerate(per_sample):
            n = min(int(p["tokens"].shape[0]), T)
            tokens[i, :n] = p["tokens"][:n]
            modality[i, :n] = p["modality"][:n]
            linear[i, :n] = p["linear"][:n]
            t[i, :n] = p["t"][:n]
            h[i, :n] = p["h"][:n]
            w[i, :n] = p["w"][:n]
            score[i, :n] = p["score"][:n]
            lengths[i] = n

        # Next-token prediction targets: shift left by 1, last column -> -100.
        targets = torch.full_like(tokens, fill_value=-100)
        targets[:, :-1] = tokens[:, 1:]
        # Score mask follows the *target* positions, not the input.
        loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
        loss_mask[:, :-1] = score[:, 1:].to(torch.float32)
        # Mask out target positions that fall on / past the sample length.
        for i in range(bsz):
            n = int(lengths[i].item())
            if n - 1 < T:
                targets[i, max(n - 1, 0):] = -100
                loss_mask[i, max(n - 1, 0):] = 0.0

        axes = RopeAxes(modality=modality, linear=linear, t=t, h=h, w=w)
        return PackedBatch(
            tokens=tokens,
            targets=targets,
            axes=axes,
            loss_mask=loss_mask,
            lengths=lengths,
        )


__all__ = ["PackedBatch", "SequencePacker"]
