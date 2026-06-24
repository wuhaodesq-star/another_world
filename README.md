# Another World

> An open, general-purpose multimodal world model.

`another_world` is an open-source effort to build a foundation world model that can
**understand**, **predict**, **imagine**, and **interact** with multimodal observations
(video / image / text / action). The project follows the staged roadmap described
in [`docs/roadmap.md`](docs/roadmap.md).

## Status

**Stage 0 — Scaffolding.** This repository currently contains:

- Project layout and packaging (`pyproject.toml`).
- Hydra configuration skeleton (`configs/`).
- A 350M toy Transformer used as a smoke test (`src/another_world/models/dynamics/toy.py`).
- Unit-test scaffolding (`tests/`).
- Docker / SLURM templates (`docker/`, `scripts/`).
- CI workflows (`.github/workflows/`).

No training weights exist yet. The full plan, including all stages 1-8 and the
6-month MVP target, lives in [`docs/roadmap.md`](docs/roadmap.md).

## Quickstart (local CPU smoke test)

```bash
# 1. Create a fresh environment (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# 2. Editable install with dev extras
pip install -e ".[dev]"

# 3. Run unit tests
pytest -q

# 4. Run the toy forward pass
python -m another_world.models.dynamics.toy
```

## Repository layout

```
another_world/
├── configs/                Hydra configs (data / model / train / eval)
├── docker/                 Training & inference container recipes
├── docs/                   Architecture, data card, model card, roadmap
├── scripts/                Launch scripts (SLURM, data prep, eval)
├── src/another_world/      Python package
│   ├── data/               Crawlers, filters, dataset loaders
│   ├── tokenizers/         Visual / action / text tokenizers
│   ├── models/             Dynamics, decoder, jepa, layers
│   ├── training/           Trainers, losses, schedulers
│   ├── eval/               FVD / VBench / Physion wrappers
│   ├── inference/          Rollout, serving
│   └── utils/              Shared utilities
└── tests/                  Unit + integration tests
```

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full multi-stage plan, the
6-month MVP definition (`another-world-7b-v0.1`), evaluation suite, and risk
register.

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
the [Code of Conduct](CODE_OF_CONDUCT.md) before opening a pull request.

## License

Licensed under the [Apache License 2.0](LICENSE).
