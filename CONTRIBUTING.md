# Contributing to Another World

Thanks for your interest! This project is in early scaffolding, but we welcome
issues, discussions, and pull requests from day one.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

## Quality checks

Every change must pass:

```bash
ruff check .
black --check .
mypy src
pytest -q
```

`pre-commit` runs the formatters automatically on each commit.

## Branching & commits

- Default branch: `main`.
- Feature branches: `feat/<scope>-<short-description>`.
- Fix branches: `fix/<scope>-<short-description>`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
  for example:
  - `feat(models): add toy 350M Transformer scaffold`
  - `fix(data): handle empty webdataset shard`
  - `chore(ci): pin ruff to 0.6.x`

## Pull request checklist

- [ ] Tests added or updated.
- [ ] `pre-commit run --all-files` passes locally.
- [ ] Docstrings updated for any public API changes.
- [ ] Linked to an issue (if applicable).

## Reporting issues

Use the GitHub issue tracker. For security or data-licensing concerns, please
open a private security advisory instead of a public issue.
