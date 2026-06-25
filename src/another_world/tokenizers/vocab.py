"""Unified multimodal vocabulary.

The dynamics model operates on a single token stream that interleaves
visual, text, and action ids. We achieve this by laying out the vocabulary
as disjoint slabs:

    [ text 0..text_size-1 ]
    [ visual text_size..text_size+visual_size-1 ]
    [ action ... ]
    [ special ... ]

This layout lets us decide modality from a token id alone, which keeps
attention masks, loss masking, and tied-embedding readout simple.

The default sizes match the stage-1 plan from ``docs/roadmap.md``:

- text:    32_000  (LLaMA-3 BPE subset)
- visual:  65_536  (Cosmos DV vocab)
- action:   4_096
- special:    256  (BOS, EOS, BOV, EOV, BOC, EOC, BOA, EOA, PAD, MASK, ...)

Total: ~101_888 ids, comfortably fits in int32.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TEXT_SIZE = 32_000
DEFAULT_VISUAL_SIZE = 65_536
DEFAULT_ACTION_SIZE = 4_096
DEFAULT_SPECIAL_SIZE = 256


SPECIAL_TOKEN_NAMES: tuple[str, ...] = (
    "pad",          # padding
    "bos",          # begin of stream
    "eos",          # end of stream
    "bov",          # begin of visual
    "eov",          # end of visual
    "boc",          # begin of caption (text)
    "eoc",          # end of caption (text)
    "boa",          # begin of action
    "eoa",          # end of action
    "mask",         # mask token for self-supervised objectives
    "sep",          # generic separator
    "unk",          # unknown
)


@dataclass(frozen=True)
class VocabLayout:
    """A frozen description of how the joint vocabulary is partitioned.

    Use :meth:`default` for the stage-1 layout, or build a custom one for
    smaller toy configurations.
    """

    text_size: int
    visual_size: int
    action_size: int
    special_size: int
    special_names: tuple[str, ...] = SPECIAL_TOKEN_NAMES

    # ----- factories ------------------------------------------------------

    @classmethod
    def default(cls) -> "VocabLayout":
        return cls(
            text_size=DEFAULT_TEXT_SIZE,
            visual_size=DEFAULT_VISUAL_SIZE,
            action_size=DEFAULT_ACTION_SIZE,
            special_size=DEFAULT_SPECIAL_SIZE,
        )

    @classmethod
    def tiny(cls) -> "VocabLayout":
        """Small layout used by toy / unit tests."""

        return cls(
            text_size=64, visual_size=128, action_size=16, special_size=16,
        )

    # ----- slab offsets ---------------------------------------------------

    @property
    def text_start(self) -> int:
        return 0

    @property
    def visual_start(self) -> int:
        return self.text_size

    @property
    def action_start(self) -> int:
        return self.text_size + self.visual_size

    @property
    def special_start(self) -> int:
        return self.text_size + self.visual_size + self.action_size

    @property
    def total_size(self) -> int:
        return (
            self.text_size + self.visual_size + self.action_size + self.special_size
        )

    # ----- id helpers -----------------------------------------------------

    def encode_text(self, local_id: int) -> int:
        if not 0 <= local_id < self.text_size:
            raise ValueError(
                f"text id {local_id} out of range [0, {self.text_size})"
            )
        return self.text_start + local_id

    def encode_visual(self, local_id: int) -> int:
        if not 0 <= local_id < self.visual_size:
            raise ValueError(
                f"visual id {local_id} out of range [0, {self.visual_size})"
            )
        return self.visual_start + local_id

    def encode_action(self, local_id: int) -> int:
        if not 0 <= local_id < self.action_size:
            raise ValueError(
                f"action id {local_id} out of range [0, {self.action_size})"
            )
        return self.action_start + local_id

    def special_id(self, name: str) -> int:
        if name not in self.special_names:
            raise KeyError(
                f"unknown special token '{name}'. "
                f"Known: {list(self.special_names)}"
            )
        return self.special_start + self.special_names.index(name)

    def modality_of(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.total_size:
            raise ValueError(f"token id {token_id} out of range")
        if token_id < self.visual_start:
            return "text"
        if token_id < self.action_start:
            return "visual"
        if token_id < self.special_start:
            return "action"
        return "special"


@dataclass
class VocabInfo:
    """Mutable companion of :class:`VocabLayout` carrying derived constants."""

    layout: VocabLayout
    pad_id: int = field(init=False)
    bos_id: int = field(init=False)
    eos_id: int = field(init=False)
    bov_id: int = field(init=False)
    eov_id: int = field(init=False)
    boc_id: int = field(init=False)
    eoc_id: int = field(init=False)
    boa_id: int = field(init=False)
    eoa_id: int = field(init=False)
    mask_id: int = field(init=False)
    sep_id: int = field(init=False)
    unk_id: int = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "pad", "bos", "eos", "bov", "eov", "boc", "eoc", "boa", "eoa",
            "mask", "sep", "unk",
        ):
            object.__setattr__(self, f"{name}_id", self.layout.special_id(name))


__all__ = [
    "DEFAULT_ACTION_SIZE",
    "DEFAULT_SPECIAL_SIZE",
    "DEFAULT_TEXT_SIZE",
    "DEFAULT_VISUAL_SIZE",
    "SPECIAL_TOKEN_NAMES",
    "VocabInfo",
    "VocabLayout",
]
