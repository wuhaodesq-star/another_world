"""Experiment logging abstraction.

Provides a small interface that the trainer can call without caring whether
W&B is installed, the user is offline, or we just want JSONL output for
CI / local debugging.

Backends
--------
- ``disabled``  : drops everything (used in tests and offline jobs).
- ``jsonl``     : appends one JSON object per call to a local file.
- ``wandb``     : forwards to ``wandb.init`` / ``wandb.log``. Falls back to
                  ``disabled`` if ``wandb`` is not installed or if
                  ``WANDB_API_KEY`` is unset.

Selection is controlled by:

- ``backend`` argument to :func:`create_logger`, or
- the ``AW_LOGGER_BACKEND`` environment variable (``disabled`` / ``jsonl`` /
  ``wandb`` / ``auto``).

``auto`` (default) picks ``wandb`` if both the package and the API key are
available; otherwise it falls back to ``jsonl``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)

LoggerBackend = str  # "disabled" | "jsonl" | "wandb" | "auto"


@runtime_checkable
class ExperimentLogger(Protocol):
    """Common interface for all logging backends."""

    backend: str

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None: ...
    def log_config(self, config: Mapping[str, Any]) -> None: ...
    def finish(self) -> None: ...


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


@dataclass
class DisabledLogger:
    backend: str = "disabled"

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        return None

    def log_config(self, config: Mapping[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


@dataclass
class JsonlLogger:
    path: Path
    backend: str = "jsonl"
    _fh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        record: dict[str, Any] = {
            "ts": time.time(),
            "step": step,
            "metrics": dict(metrics),
        }
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def log_config(self, config: Mapping[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "config": dict(config),
        }
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def finish(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()


@dataclass
class WandbLogger:
    project: str
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    mode: str = "online"  # online | offline | disabled
    backend: str = "wandb"
    _run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - only when wandb missing
            raise ImportError(
                "wandb is not installed. Install with `pip install wandb` or "
                "use backend='jsonl'."
            ) from exc

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            tags=self.tags,
            config=dict(self.config),
            mode=self.mode,
            reinit=True,
        )

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        import wandb  # type: ignore[import-not-found]

        if step is None:
            wandb.log(dict(metrics))
        else:
            wandb.log(dict(metrics), step=step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        import wandb  # type: ignore[import-not-found]

        wandb.config.update(dict(config), allow_val_change=True)

    def finish(self) -> None:
        import wandb  # type: ignore[import-not-found]

        if self._run is not None:
            wandb.finish()
            self._run = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _wandb_available() -> bool:
    try:
        import wandb  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend(backend: LoggerBackend) -> LoggerBackend:
    if backend != "auto":
        return backend
    if _wandb_available() and os.environ.get("WANDB_API_KEY"):
        return "wandb"
    return "jsonl"


def create_logger(
    backend: LoggerBackend = "auto",
    *,
    project: str = "another_world",
    entity: str | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    config: Mapping[str, Any] | None = None,
    jsonl_path: str | os.PathLike[str] | None = None,
    wandb_mode: str = "online",
) -> ExperimentLogger:
    """Create an :class:`ExperimentLogger` for the chosen backend.

    Args:
        backend: ``disabled`` / ``jsonl`` / ``wandb`` / ``auto``. When ``auto``,
            wandb is selected if installed and ``WANDB_API_KEY`` is set,
            otherwise jsonl.
        project / entity / run_name / tags / config: forwarded to wandb.
        jsonl_path: file used by the jsonl backend
            (default ``outputs/logs/run.jsonl``).
        wandb_mode: forwarded to ``wandb.init`` (``online`` / ``offline`` /
            ``disabled``).
    """

    backend = os.environ.get("AW_LOGGER_BACKEND", backend)  # type: ignore[assignment]
    backend = _resolve_backend(backend)

    if backend == "disabled":
        _LOG.info("ExperimentLogger backend: disabled")
        return DisabledLogger()

    if backend == "jsonl":
        path = Path(jsonl_path) if jsonl_path else Path("outputs/logs/run.jsonl")
        _LOG.info("ExperimentLogger backend: jsonl -> %s", path)
        logger = JsonlLogger(path=path)
        if config is not None:
            logger.log_config(config)
        return logger

    if backend == "wandb":
        if not _wandb_available():
            _LOG.warning("wandb requested but not installed; falling back to jsonl.")
            return create_logger(
                "jsonl",
                project=project,
                entity=entity,
                run_name=run_name,
                tags=tags,
                config=config,
                jsonl_path=jsonl_path,
            )
        _LOG.info("ExperimentLogger backend: wandb (project=%s)", project)
        return WandbLogger(
            project=project,
            entity=entity,
            run_name=run_name,
            tags=list(tags or []),
            config=dict(config or {}),
            mode=wandb_mode,
        )

    raise ValueError(
        f"unknown logger backend '{backend}'; "
        "expected one of: disabled, jsonl, wandb, auto"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):  # torch / numpy scalars
        try:
            return obj.item()
        except Exception:  # pragma: no cover
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


__all__ = [
    "DisabledLogger",
    "ExperimentLogger",
    "JsonlLogger",
    "LoggerBackend",
    "WandbLogger",
    "create_logger",
]
