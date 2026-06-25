"""Tests for the multimodal trainer."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout
from another_world.training.multimodal import (
    MultimodalTrainerConfig,
    apply_activation_checkpointing,
    build_optimizer,
    run_multimodal_training,
)


def _packed_batches(layout, packer, n_batches: int, batch_size: int = 2):
    samples = []
    for i in range(n_batches * batch_size):
        samples.append(
            TokenSample(
                visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
                text_tokens=torch.tensor([i % 4, (i + 1) % 4], dtype=torch.long),
                key=f"k{i}",
            )
        )
    batches = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        batches.append(packer.pack_batch(chunk))
    return batches


def test_multimodal_training_runs_and_logs() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32)
    cfg = MultimodalDynamicsConfig.toy(layout.total_size)
    model = MultimodalDynamicsModel(cfg)
    batches = _packed_batches(layout, packer, n_batches=8, batch_size=2)
    history = run_multimodal_training(
        model,
        batches,
        MultimodalTrainerConfig(
            steps=20, lr=5e-3, warmup_steps=2, log_every=5,
            device="cpu", precision="fp32", seed=0,
        ),
    )
    assert len(history) >= 2
    assert history[0].loss > history[-1].loss


def test_grad_accum_works_and_uses_multiple_batches() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    batches = iter(_packed_batches(layout, packer, n_batches=20, batch_size=2))

    consumed = 0
    def _counting():
        nonlocal consumed
        for b in batches:
            consumed += 1
            yield b

    history = run_multimodal_training(
        model, _counting(),
        MultimodalTrainerConfig(
            steps=4, grad_accum=3, log_every=1, lr=1e-3, warmup_steps=1,
            device="cpu", precision="fp32", seed=0,
        ),
    )
    assert len(history) == 4
    # 4 steps * 3 micro-batches each = 12 packed batches must be consumed.
    assert consumed >= 12


def test_activation_checkpointing_is_idempotent() -> None:
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    apply_activation_checkpointing(model)
    apply_activation_checkpointing(model)  # second call should be a no-op
    for block in model.layers:
        assert getattr(block, "_ac_wrapped", False) is True


def test_activation_checkpointing_keeps_backward() -> None:
    """Training with checkpointing must still produce gradients."""

    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    batches = _packed_batches(layout, packer, n_batches=4, batch_size=2)
    run_multimodal_training(
        model, batches,
        MultimodalTrainerConfig(
            steps=4, lr=1e-3, warmup_steps=1, log_every=1,
            activation_checkpointing=True,
            device="cpu", precision="fp32", seed=0,
        ),
    )
    grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert grad_seen


def test_build_optimizer_separates_decay_groups() -> None:
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    optim = build_optimizer(
        model,
        MultimodalTrainerConfig(weight_decay=0.05),
        fused=False,
    )
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.05
    assert optim.param_groups[1]["weight_decay"] == 0.0
    # ``norm`` weights are 1-D so they must live in the no-decay group.
    norm_params = {id(model.norm.weight)}
    nodecay_ids = {id(p) for p in optim.param_groups[1]["params"]}
    assert norm_params.issubset(nodecay_ids)
