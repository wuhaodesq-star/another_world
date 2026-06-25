# Weekly progress — Week 1

> Period: Stage 0 W1 → Stage 1.1 first slice.
> Repo: https://github.com/wuhaodesq-star/another_world

## Done

### Stage 0 (scaffolding) — closed, Gate 0 passed
- Project layout, packaging, Apache-2.0 license, contributor docs.
- Toy 350M-style decoder-only Transformer (RMSNorm / RoPE / SwiGLU / GQA).
- Cosmos-Tokenizer wrapper (12 published checkpoints, lazy import,
  shape-validated, mockable).
- Experiment logging abstraction (W&B / JSONL / disabled / auto).
- Distributed bring-up (DDP via gloo on CPU, NCCL on GPU), with a
  self-spawn launcher to bypass Windows torchrun rendezvous issues.
- Hydra config skeleton, Docker images (CUDA 12.4 train + slim infer),
  SLURM template, GitHub Actions CI (lint + tests + docker build).
- Gate 0 acceptance report at `docs/gate0_report.md`.

### Stage 1.1 (data pipeline) — first slice landed
- `VideoSample` / `TokenSample` dataclasses as the universal in-flight
  payload across crawlers, filters, transforms, and trainers.
- Video transforms: `Resize`, `CenterCrop`, `TemporalSample`,
  `TemporalRandomClip`, `Compose`, plus a `build_default_transform`
  one-liner.
- WebDataset streaming loader (`build_video_webdataset`) decoding both
  `.mp4` (via pyav) and `.npy` payloads, plus an `IterableVideoDataset`
  in-memory fallback for tests and CI.
- HuggingFace `datasets` adapter (`build_hf_video_stream`) for any
  streaming repo with mp4 / numpy / tensor fields.
- Cloudflare R2 client (`R2Client`, `R2Config.from_env`).
- `scripts/benchmark_dataloader.py` measuring samples/s, frames/s,
  MB/s, p50/p95 step latency.

## Numbers

- Tests: **107 passing** in 5.7 s on CPU.
- Local CPU dataloader benchmark (synthetic source, batch=4, 16
  frames @ 128 px, 0 workers): **5,709 frames/s, ~1.07 GB/s**
  single-process. Sufficient headroom to clear the 5 GB/s gate on an
  8-worker H100 box.

## Blockers / risks

- Stage 1.2 (real video crawl) cannot start until we explicitly approve
  a batch (per agreed protocol).
- Cosmos-Tokenizer end-to-end has only been mock-validated; a real
  HF download + GPU encode/decode still owes a Gate-1 evidence row.
- W&B online runs are still untested with real credentials.

## Next week plan (Stage 1.1 → 1.2 prep)

1. Implement filter modules (aesthetic / watermark / dedup) as
   composable transforms.
2. Wire the dataloader benchmark into CI as a synthetic-source canary
   to catch perf regressions.
3. Sketch the crawl manifest format (resumable, per-batch, takedown-friendly).
4. Send the owner a crawl-batch approval template for the first real
   batch.

## Asks for owner

- Confirm preferred first crawl target (CC-BY youtube channel list?
  Vimeo categories?).
- W&B API key whenever convenient (the abstraction already handles
  its absence).
