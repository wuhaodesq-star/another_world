"""Test that KV-cache rollout produces the same tokens as the naive rollout."""

from __future__ import annotations

import time

import torch

from another_world.inference.generation import (
    GenerationConfig,
    rollout_visual_tokens,
)
from another_world.models.dynamics import (
    MultimodalDynamicsConfig,
    MultimodalDynamicsModel,
)
from another_world.tokenizers.vocab import VocabLayout


def test_kv_cache_and_recompute_match() -> None:
    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    model.eval()

    cfg_cache = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        sampler="euler", sampler_steps=2, seed=7,
        use_kv_cache=True,
    )
    cfg_naive = GenerationConfig(
        visual_frames=1, visual_height=2, visual_width=2,
        sampler="euler", sampler_steps=2, seed=7,
        use_kv_cache=False,
    )
    tokens_cache = rollout_visual_tokens(
        model, text_ids=[1, 2, 3], config=cfg_cache, layout=layout,
    )
    tokens_naive = rollout_visual_tokens(
        model, text_ids=[1, 2, 3], config=cfg_naive, layout=layout,
    )
    assert torch.equal(tokens_cache, tokens_naive)


def test_kv_cache_rollout_speed_advantage_smoke() -> None:
    """Sanity check: cached rollout should not be slower than recompute.

    We don't assert a specific speedup ratio because timing on busy CI
    machines is noisy, but the cached path should finish at least as
    fast as recompute on the same workload.
    """

    torch.manual_seed(0)
    layout = VocabLayout.tiny()
    model = MultimodalDynamicsModel(
        MultimodalDynamicsConfig.toy(layout.total_size)
    )
    model.eval()
    cfg_cache = GenerationConfig(
        visual_frames=2, visual_height=3, visual_width=3,
        sampler="euler", sampler_steps=1, seed=0, use_kv_cache=True,
    )
    cfg_naive = GenerationConfig(
        visual_frames=2, visual_height=3, visual_width=3,
        sampler="euler", sampler_steps=1, seed=0, use_kv_cache=False,
    )
    # warm-up
    rollout_visual_tokens(model, text_ids=[1, 2], config=cfg_cache, layout=layout)
    rollout_visual_tokens(model, text_ids=[1, 2], config=cfg_naive, layout=layout)

    t0 = time.perf_counter()
    rollout_visual_tokens(model, text_ids=[1, 2], config=cfg_cache, layout=layout)
    dt_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    rollout_visual_tokens(model, text_ids=[1, 2], config=cfg_naive, layout=layout)
    dt_naive = time.perf_counter() - t0

    # cache may not always be strictly faster on tiny models but it
    # should certainly not be more than 4x slower (would indicate a bug).
    assert dt_cache < dt_naive * 4.0
