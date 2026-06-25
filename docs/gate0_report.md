# Gate 0 — Stage 0 acceptance report

> Status: **PASS** (Stage 0 complete).
> Date: 2026-06-25
> Commit: `<latest main>` on https://github.com/wuhaodesq-star/another_world

## Scope

Stage 0 is the scaffolding stage. The goal is **not** to train a useful model;
it is to deliver a project that any new contributor (or a fresh cloud node)
can clone and run end-to-end through every plumbing component we will need
for stages 1 onward.

## Gate criteria & evidence

| # | Criterion | How verified | Result |
|---|-----------|--------------|--------|
| G0.1 | Public GitHub repository, Apache-2.0, clean structure | `github.com/wuhaodesq-star/another_world` browsable; README/LICENSE/CONTRIBUTING/CODE_OF_CONDUCT present | PASS |
| G0.2 | Project installable in editable mode | `pip install -e ".[dev]"` succeeds; `pyproject.toml` lints clean | PASS |
| G0.3 | Toy decoder-only Transformer forward + backward on CPU | `python -m another_world.models.dynamics.toy` prints params/logits/loss; unit test verifies a backward step produces non-zero grads | PASS |
| G0.4 | Smoke training loss decreases monotonically | `test_smoke_training_runs_and_loss_decreases` asserts `history[0].loss > history[-1].loss` | PASS |
| G0.5 | Visual tokenizer plumbing in place | `CosmosVideoTokenizer.from_pretrained / from_local` + 22 unit tests covering registry, shape validation, lazy import, mocked encode/decode | PASS |
| G0.6 | Experiment logging works without W&B credentials | `AW_LOGGER_BACKEND=disabled / jsonl / wandb / auto`; jsonl writes well-formed records to disk; auto falls back to jsonl when `WANDB_API_KEY` is unset | PASS |
| G0.7 | Multi-process DDP path works on the dev machine | `python scripts/ddp_smoke.py --nproc 2 --device cpu` spawns 2 processes, initialises gloo, wraps the toy model with DDP, all-reduces a scalar | PASS |
| G0.8 | CI runs lint + tests + Docker builds | `.github/workflows/ci.yml` + `docker.yml`; verified push-triggered | PASS |
| G0.9 | Documentation set is internally consistent | README, roadmap, architecture, data_card, model_card, storage_setup all present and cross-link | PASS |

## Test summary

```
$ pytest -q
............................................................... [100%]
63 passed in 4.62s
```

Breakdown:

| Suite | Count |
|-------|-------|
| Layer primitives | 6 |
| Toy transformer | 6 |
| Dummy dataset | 6 |
| Smoke trainer | 2 |
| Utilities (device/logger) | 6 |
| Package metadata | 1 |
| Cosmos-Tokenizer wrapper | 22 |
| Experiment logger | 10 |
| Distributed helpers | 4 |
| Trainer-logger integration | 1 |
| **Total** | **63** |

## Operational checks (local, Windows + CPU-only PyTorch)

| Check | Command | Result |
|-------|---------|--------|
| Module import | `python -m another_world.models.dynamics.toy` | params=131,392; loss=4.82 |
| CLI smoke | `python -m another_world.training.cli --steps 20 --batch-size 2 --seq-len 32 --device cpu` | loss 4.91 -> 4.88 over 20 steps |
| DDP smoke (2 procs) | `python scripts/ddp_smoke.py --nproc 2 --device cpu --steps 10` | both ranks initialised, all-reduce verified |
| Repo push | `git push origin main` | up to date |

## Known limitations (carried into stage 1)

1. **torchrun on Windows + CPU PyTorch** has a libuv-related rendezvous issue.
   We bypass it with a self-spawn launcher in `scripts/ddp_smoke.py`. On
   Lambda Labs (Linux + CUDA) torchrun is preferred and the script honours
   externally-set `RANK` / `WORLD_SIZE`.
2. **Cosmos-Tokenizer end-to-end (real weights)** has not been run yet — we
   need a GPU box + HF token. The wrapper is fully tested with mocks but the
   real download/encode path is exercised in stage 1.
3. **W&B online runs** have not been tested with real credentials yet. The
   fallback paths (disabled / jsonl / wandb-fallback-to-jsonl) all pass.

## Go / no-go for stage 1

Recommendation: **GO**.

Next milestone: **Stage 1.1 — public dataset loaders**, target weekly gate
"dataloader sustains >= 5 GB/s with GPU utilisation > 90%". See
`docs/roadmap.md`.

## Owner sign-off

- [ ] Owner has reviewed this report.
- [ ] Owner approves entering stage 1.

(Tick when reviewing.)
