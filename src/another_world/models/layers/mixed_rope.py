"""Mixed positional encoding for multimodal token streams.

A single packed sequence may contain text tokens, visual tokens, and action
tokens. To use the same attention kernel for all of them we precompute
rotary frequencies *per position* based on each token's modality and its
local (t, h, w) coordinates within a visual block.

Layout used by :class:`MixedRoPE`
---------------------------------

For each token in the sequence we record:

- modality           : 0=text, 1=visual, 2=action, 3=special
- linear_position    : monotonically increasing index in the stream
- t / h / w          : visual cube coordinates (0 for non-visual tokens)

The head dim ``D`` is split four ways for visual tokens, ``[D/4]`` each for
``linear``, ``t``, ``h``, ``w``. For non-visual tokens all four shards
encode the same ``linear_position`` so the effective encoding degenerates
to standard RoPE-1D. This keeps the attention kernel modality-agnostic.

Reference: "RoPE-Tie" / "M-RoPE" used in InternVL / Qwen2-VL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RopeAxes:
    """Per-token coordinate tensors with shape ``[B, T]`` (long)."""

    modality: Tensor
    linear: Tensor
    t: Tensor
    h: Tensor
    w: Tensor

    def to(self, device: torch.device) -> "RopeAxes":
        return RopeAxes(
            modality=self.modality.to(device),
            linear=self.linear.to(device),
            t=self.t.to(device),
            h=self.h.to(device),
            w=self.w.to(device),
        )


class MixedRoPE:
    """Compute per-token (cos, sin) tables for mixed-modality RoPE.

    The total head dimension is split into four equal shards. Each shard
    is a regular RoPE-1D table addressed by a different axis:

    - shard 0: linear position
    - shard 1: t (time)
    - shard 2: h (height)
    - shard 3: w (width)

    For non-visual tokens, t / h / w fall back to ``linear`` so all four
    shards encode the same scalar.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        max_linear: int = 65_536,
        max_t: int = 256,
        max_h: int = 256,
        max_w: int = 256,
        theta: float = 10_000.0,
    ) -> None:
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim must be a multiple of 8 (got {head_dim}); "
                "we split into 4 RoPE shards of even size each."
            )
        self.head_dim = head_dim
        self.shard_dim = head_dim // 4
        if self.shard_dim % 2 != 0:
            raise ValueError("shard_dim must be even for RoPE")

        self._tables = {
            "linear": _make_table(max_linear, self.shard_dim, theta),
            "t": _make_table(max_t, self.shard_dim, theta),
            "h": _make_table(max_h, self.shard_dim, theta),
            "w": _make_table(max_w, self.shard_dim, theta),
        }

    # ----- public API -----------------------------------------------------

    def build(
        self,
        axes: RopeAxes,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` with shape ``[B, T, head_dim // 2]``."""

        if device is None:
            device = axes.linear.device

        # For non-visual tokens (modality != 1) we fall back to linear pos.
        visual_mask = (axes.modality == 1).to(torch.long)
        t = torch.where(visual_mask.bool(), axes.t, axes.linear)
        h = torch.where(visual_mask.bool(), axes.h, axes.linear)
        w = torch.where(visual_mask.bool(), axes.w, axes.linear)

        cos_lin, sin_lin = self._lookup("linear", axes.linear, device, dtype)
        cos_t, sin_t = self._lookup("t", t, device, dtype)
        cos_h, sin_h = self._lookup("h", h, device, dtype)
        cos_w, sin_w = self._lookup("w", w, device, dtype)

        cos = torch.cat([cos_lin, cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_lin, sin_t, sin_h, sin_w], dim=-1)
        return cos, sin

    def _lookup(
        self,
        name: str,
        coords: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        cos_table, sin_table = self._tables[name]
        cos_table = cos_table.to(device=device, dtype=dtype)
        sin_table = sin_table.to(device=device, dtype=dtype)
        clamped = coords.clamp_max(cos_table.size(0) - 1).long()
        cos = cos_table.index_select(0, clamped.reshape(-1))
        sin = sin_table.index_select(0, clamped.reshape(-1))
        cos = cos.view(*coords.shape, -1)
        sin = sin.view(*coords.shape, -1)
        return cos, sin


def _make_table(length: int, shard_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    half = shard_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(length, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


# ---------------------------------------------------------------------------
# Helpers to build axes for typical sequence layouts
# ---------------------------------------------------------------------------


def axes_from_segments(
    segments: list[tuple[str, dict]],
    *,
    device: torch.device | None = None,
) -> RopeAxes:
    """Build :class:`RopeAxes` for a single sample from a segment description.

    Each segment is ``(modality, params)`` where modality is one of
    ``"text" / "visual" / "action" / "special"``. For ``"visual"`` we
    expect ``{"t": int, "h": int, "w": int}`` to construct THW coords.

    Returns ``RopeAxes`` with shape ``[1, T]``.
    """

    mod_codes = {"text": 0, "visual": 1, "action": 2, "special": 3}
    modality: list[int] = []
    t_coords: list[int] = []
    h_coords: list[int] = []
    w_coords: list[int] = []
    linear: list[int] = []
    pos = 0
    for mod, params in segments:
        code = mod_codes[mod]
        if mod == "visual":
            t_dim, h_dim, w_dim = params["t"], params["h"], params["w"]
            for ti in range(t_dim):
                for hi in range(h_dim):
                    for wi in range(w_dim):
                        modality.append(code)
                        t_coords.append(ti)
                        h_coords.append(hi)
                        w_coords.append(wi)
                        linear.append(pos)
                        pos += 1
        else:
            count = params.get("count", params.get("length", 0))
            for _ in range(count):
                modality.append(code)
                t_coords.append(0)
                h_coords.append(0)
                w_coords.append(0)
                linear.append(pos)
                pos += 1

    def _t(values: list[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)

    return RopeAxes(
        modality=_t(modality),
        linear=_t(linear),
        t=_t(t_coords),
        h=_t(h_coords),
        w=_t(w_coords),
    )


__all__ = ["MixedRoPE", "RopeAxes", "axes_from_segments"]
