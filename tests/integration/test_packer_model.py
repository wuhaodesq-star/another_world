"""End-to-end test: TokenSample -> SequencePacker -> MultimodalDynamicsModel."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.data.datasets.sequence_packer import SequencePacker
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def test_packer_to_model_forward_and_backward() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=32)
    cfg = MultimodalDynamicsConfig.toy(layout.total_size)
    model = MultimodalDynamicsModel(cfg)

    samples = [
        TokenSample(
            visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
            text_tokens=torch.tensor([3, 5], dtype=torch.long),
            key=f"k{i}",
        )
        for i in range(2)
    ]
    batch = packer.pack_batch(samples)
    out = model(
        batch.tokens,
        axes=batch.axes,
        targets=batch.targets,
        loss_mask=batch.loss_mask,
    )
    assert out["logits"].shape == (2, 32, layout.total_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    # Verify at least the embedding got a gradient (closest to inputs).
    assert model.tok_embeddings.weight.grad is not None
    assert model.tok_embeddings.weight.grad.abs().sum().item() > 0


def test_packer_to_model_loss_decreases_overfit() -> None:
    """Optimise a tiny model on a single batch; loss must drop."""

    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    packer = SequencePacker(layout, max_len=24)
    cfg = MultimodalDynamicsConfig.toy(layout.total_size)
    model = MultimodalDynamicsModel(cfg)

    samples = [
        TokenSample(
            visual_tokens=torch.randint(0, 16, (1, 2, 2), dtype=torch.long),
            text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
        )
    ]
    batch = packer.pack_batch(samples)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    initial_loss = float(
        model(batch.tokens, axes=batch.axes, targets=batch.targets,
              loss_mask=batch.loss_mask)["loss"].item()
    )
    for _ in range(40):
        optim.zero_grad(set_to_none=True)
        out = model(
            batch.tokens, axes=batch.axes,
            targets=batch.targets, loss_mask=batch.loss_mask,
        )
        out["loss"].backward()
        optim.step()
    final_loss = float(
        model(batch.tokens, axes=batch.axes, targets=batch.targets,
              loss_mask=batch.loss_mask)["loss"].item()
    )
    assert final_loss < initial_loss
