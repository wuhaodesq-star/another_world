# Another World — Roadmap

> Locked plan (v2.0). Status: **Stage 0 — Scaffolding (in progress)**.

## Mission

Build an open, general-purpose multimodal world model that can:

1. **Understand** scene semantics from video / image / text / action streams.
2. **Predict** future multimodal states conditioned on past observations,
   text instructions, and/or actions.
3. **Imagine** counterfactual rollouts in latent space.
4. **Interact** with users via language and actions.
5. **Generalise** across domains (driving, ego-centric, gaming, robotics).

## Locked decisions

| # | Topic | Value |
|---|-------|-------|
| 1 | Compute | Cloud rental on **Lambda Labs**, 8x H100 single node to start |
| 2 | Visual tokenizer | **Open-source** to start (Cosmos-Tokenizer CV8x8x8), revisit later |
| 3 | Training framework | **TorchTitan** primary; Megatron-LM as a fallback for MoE |
| 4 | Frameworks | **PyTorch** primary; **JAX** added in late stages |
| 5 | 6-month MVP | **(a)** 7B model: 1 first-frame + text -> 5s 512x288 24fps video |
| 6 | Data crawling | Allowed. **Owner is notified before every crawl batch** |
| 7 | Open source | **Public from day one**, Apache-2.0 |
| 8 | Storage | **Cloudflare R2** object storage |
| 9 | Human MOS | Owner + automated scoring |
| 10 | Reporting | Weekly written report + W&B link; per-gate detailed report |
| 11 | Stop signal | A single "pause" message halts work and snapshots state |

## Architecture overview

```
[video / image / text / action]
        |
        v
[modality-specific tokenizers]
        |
        v
[central dynamics Transformer (decoder-only, 7B/30B/70B)]
        |
        v
[per-modality decoders: DiT for pixels, LM head for text, regression heads
 for physical quantities / rewards / termination]
```

## Stage map

| Stage | Window | Goal | Gate |
|-------|--------|------|------|
| 0 | M1 W1-2 | Scaffolding, Docker, SLURM, W&B, 8-GPU smoke test | 350M toy converges in 100 steps |
| 1.1 | M1 W3-4 | Public dataset loaders (>=5 GB/s) | Dataloader not the bottleneck (GPU > 90%) |
| 1.2 | M2 W5-7 | Crawl + scene split + ASR + caption pipeline | 100k clean CC videos tokenised |
| 1.3 | M2 W8 | Cosmos pre-tokenisation -> WebDataset shards | >= 300B total tokens |
| 3.1 | M3 W9-12 | TorchTitan dynamics model implementation | 1B subset converges cleanly |
| 3.2 | M4 W13-16 | 7B pretraining on ~1T tokens (256 H100) | 16-frame FVD < 250 (latent) |
| 4 | M5 W17-20 | DiT decoder, token -> high-res pixels | 5s 512p video successfully generated |
| 5 | M5 W21 | Evaluation suite (VBench + long-horizon + human) | v0.1 candidate is benchmarked |
| 6.2 / 8.4 | M6 W22-24 | Instruction tune + Gradio demo | MVP targets met -> release v0.1 |

## 6-month MVP definition (`another-world-7b-v0.1`)

**Inputs**:

- 1 reference image (first frame).
- Natural language prompt (<= 77 tokens).

**Outputs**:

- 5 seconds of video at 512x288, 24 fps (120 frames).

**Pass criteria**:

- VBench total score >= 70.
- Human MOS >= 3.5 / 5.
- Single-H100 inference <= 60s per clip.

**Out of scope for v0.1**:

- Video > 10 seconds.
- Multi-scene transitions.
- Interactive action control (v0.2).
- Physics-grade consistency (v0.3).

## Evaluation matrix (stage 5)

| Dimension | Metric(s) | Dataset(s) |
|-----------|-----------|-----------|
| Video quality | FVD, IS, CLIPSIM | UCF-101, MSR-VTT |
| Long-horizon | 16/64/256-step error | Internal long-horizon set |
| Physics | Physion / PhyWorld | Physion-v2 |
| Action control | Action-trajectory alignment | RT-X, Bridge |
| Text control | T2V-CompBench, VBench | VBench |
| Counterfactual | Internal QA | Internal |
| Cross-domain | Zero-shot | Held-out CARLA / Ego4D |

## Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Data licensing dispute | High | Strict CC filter; takedown workflow; synthetic data |
| Tokenizer training fails | High | Default to open-source weights |
| Loss spikes at 70B | Medium | muP, grad clip, automatic rollback |
| Long-horizon drift | High | V-JEPA latent loss + rollout consistency |
| Compute outage | Medium | Multi-cloud backups; 30-min checkpoint cadence |
| Eval disagreement | Medium | VBench + human MOS + public demos |

## Reporting cadence

- **Weekly** written summary including: completed items, blockers, next-week plan,
  W&B run link, current burn-rate.
- **Per-gate** detailed report with metrics, samples, and a go/no-go recommendation.
- **Crawl-batch approval** required before each crawl batch.
- **Pause** at any time halts execution and snapshots state.
