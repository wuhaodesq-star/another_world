"""Tests for the V-JEPA auxiliary predictor and EMA shadow."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from another_world.models.jepa import (
    EmaShadow,
    JEPAConfig,
    JEPALatentPredictor,
    jepa_loss,
)


def test_predictor_forward_shape() -> None:
    cfg = JEPAConfig(in_dim=32, out_dim=24, hidden_dim=48, n_layers=2, n_heads=4)
    pred = JEPALatentPredictor(cfg)
    x = torch.randn(2, 6, 32)
    out = pred(x)
    assert out.shape == (2, 6, 24)


def test_predictor_rejects_wrong_dim() -> None:
    pred = JEPALatentPredictor(JEPAConfig(in_dim=16, out_dim=16))
    with pytest.raises(ValueError):
        pred(torch.randn(2, 6))


def test_jepa_loss_finite_and_metrics() -> None:
    torch.manual_seed(0)
    cfg = JEPAConfig(in_dim=16, out_dim=16, hidden_dim=32)
    pred = JEPALatentPredictor(cfg)
    student = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    loss, m = jepa_loss(pred, student_hidden=student, target_hidden=target)
    assert torch.isfinite(loss)
    assert "jepa_cos" in m
    assert -1.0 <= m["jepa_cos"] <= 1.0


def test_jepa_loss_zero_with_mask_zero() -> None:
    cfg = JEPAConfig(in_dim=16, out_dim=16, hidden_dim=32)
    pred = JEPALatentPredictor(cfg)
    student = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    mask = torch.zeros(2, 4)
    loss, _ = jepa_loss(pred, student_hidden=student, target_hidden=target, mask=mask)
    # The mean over a zero mask returns 0 (denominator clamped to 1).
    assert float(loss) == 0.0


def test_jepa_loss_gradient_flows_through_predictor_only() -> None:
    """The loss must produce gradients on the predictor but not on the target."""
    cfg = JEPAConfig(in_dim=16, out_dim=16, hidden_dim=32)
    pred = JEPALatentPredictor(cfg)
    student = torch.randn(2, 4, 16, requires_grad=True)
    target = torch.randn(2, 4, 16, requires_grad=True)
    loss, _ = jepa_loss(pred, student_hidden=student, target_hidden=target)
    loss.backward()
    assert any(p.grad is not None for p in pred.parameters())
    # Student receives gradients (it's not detached); target is detached.
    assert student.grad is not None
    assert target.grad is None


def test_jepa_loss_shape_mismatch_raises() -> None:
    pred = JEPALatentPredictor(JEPAConfig(in_dim=16, out_dim=16))
    with pytest.raises(ValueError):
        jepa_loss(
            pred,
            student_hidden=torch.randn(2, 4, 16),
            target_hidden=torch.randn(2, 5, 16),
        )


# ---------------------------------------------------------------------------
# EMA shadow
# ---------------------------------------------------------------------------


def test_ema_shadow_decay_validation() -> None:
    mod = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        EmaShadow(mod, decay=0.0)
    with pytest.raises(ValueError):
        EmaShadow(mod, decay=1.5)


def test_ema_shadow_update_changes_slowly() -> None:
    torch.manual_seed(0)
    mod = nn.Linear(4, 4)
    ema = EmaShadow(mod, decay=0.9)
    # Save the initial shadow state.
    initial = {k: v.clone() for k, v in ema._shadow.items()}

    # Drastically change the module.
    with torch.no_grad():
        for p in mod.parameters():
            p.fill_(100.0)
    ema.update(mod)

    # Each shadow weight should equal decay * old + 0.1 * 100.
    for name, before in initial.items():
        expected = 0.9 * before + 0.1 * 100.0
        assert torch.allclose(ema._shadow[name], expected)


def test_ema_shadow_copy_to_round_trip() -> None:
    src = nn.Linear(4, 4)
    dst = nn.Linear(4, 4)
    # Drastically perturb dst.
    with torch.no_grad():
        for p in dst.parameters():
            p.fill_(7.0)
    ema = EmaShadow(src, decay=0.5)
    ema.copy_to(dst)
    for (k_src, p_src), (_k_dst, p_dst) in zip(
        src.named_parameters(), dst.named_parameters()
    ):
        assert torch.equal(p_src, p_dst), f"mismatch on {k_src}"


def test_ema_shadow_rejects_different_module() -> None:
    a = nn.Linear(4, 4)
    b = nn.Linear(4, 4)
    ema = EmaShadow(a, decay=0.5)
    with pytest.raises(RuntimeError):
        ema.update(b)
