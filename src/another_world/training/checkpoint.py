"""Checkpoint save / load for the multimodal trainer.

A checkpoint is a directory containing:

    checkpoint/
    +- model.safetensors   (or model.pt as fallback)
    +- optimizer.pt
    +- meta.json           (step, lr, config, timestamps, git hash)

We use ``safetensors`` when available because it is zero-copy mmappable and
safe to load with ``weights_only=True``; otherwise we fall back to
``torch.save``.  The optimizer state is always ``torch.save`` since it
contains Python dicts that ``safetensors`` cannot represent.

Distributed semantics
---------------------
Only **rank 0** is allowed to write a checkpoint. ``save_checkpoint``
gathers a full state dict on rank 0 first (for FSDP this requires
``StateDictOptions(full_state_dict=True)``); for non-FSDP wrapped models
we just call ``model.state_dict()``.

``load_checkpoint`` is symmetrical: rank 0 reads the file, broadcasts the
state dict, then every rank loads.

Optional R2 upload is handled by the ``upload_uri`` argument; if it
starts with ``s3://`` or ``r2://`` we treat it as ``<bucket>/<prefix>``
and push the entire checkpoint directory via :class:`R2Client`.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class CheckpointMeta:
    step: int
    config: dict[str, Any] = field(default_factory=dict)
    lr: float | None = None
    epoch: int | None = None
    timestamp: float = field(default_factory=time.time)
    git_hash: str | None = None
    notes: str | None = None
    framework: str = "pytorch"
    model_class: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text: str) -> "CheckpointMeta":
        return cls(**json.loads(text))


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


MODEL_PT = "model.pt"
MODEL_SAFETENSORS = "model.safetensors"
OPTIMIZER_PT = "optimizer.pt"
META_JSON = "meta.json"


def _try_safetensors() -> bool:
    try:
        import safetensors.torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def _strip_fsdp_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Drop a leading ``module.`` prefix (DDP / FSDP wrap)."""

    if all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def save_checkpoint(
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    meta: CheckpointMeta,
    is_main: bool = True,
    prefer_safetensors: bool = True,
    upload_uri: str | None = None,
) -> Path:
    """Persist ``model`` (and optionally ``optimizer``) to ``directory``.

    Only the process for which ``is_main`` is True does any I/O. Returns
    the directory path on rank 0; other ranks get the same path but do
    not touch the filesystem.
    """

    directory = Path(directory)
    if not is_main:
        return directory

    tmp = directory.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    state = _strip_fsdp_keys(model.state_dict())

    used_safetensors = False
    if prefer_safetensors and _try_safetensors():
        try:
            from safetensors.torch import save_file  # type: ignore[import-not-found]

            save_file(state, str(tmp / MODEL_SAFETENSORS))
            used_safetensors = True
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("safetensors save failed (%s); falling back to torch.save", exc)
    if not used_safetensors:
        torch.save(state, tmp / MODEL_PT)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / OPTIMIZER_PT)

    meta.framework = "pytorch"
    meta.model_class = type(model).__name__
    (tmp / META_JSON).write_text(meta.to_json(), encoding="utf-8")

    # Atomic rename (best effort).
    if directory.exists():
        shutil.rmtree(directory)
    tmp.rename(directory)

    _LOG.info("Saved checkpoint to %s (step=%d)", directory, meta.step)

    if upload_uri:
        _upload_directory(directory, upload_uri)

    return directory


def load_checkpoint(
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> CheckpointMeta:
    """Load ``model`` (and optionally ``optimizer``) from ``directory``."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")

    meta_path = directory / META_JSON
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {META_JSON} in {directory}")
    meta = CheckpointMeta.from_json(meta_path.read_text(encoding="utf-8"))

    safetensors_path = directory / MODEL_SAFETENSORS
    pt_path = directory / MODEL_PT
    if safetensors_path.exists() and _try_safetensors():
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        state = load_file(str(safetensors_path), device=str(map_location or "cpu"))
    elif pt_path.exists():
        state = torch.load(pt_path, map_location=map_location, weights_only=False)
    else:
        raise FileNotFoundError(
            f"neither {MODEL_SAFETENSORS} nor {MODEL_PT} found in {directory}"
        )

    # Tolerate wrapped state dicts.
    state = _strip_fsdp_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        _LOG.warning("missing keys when loading: %s", missing)
    if unexpected:
        _LOG.warning("unexpected keys when loading: %s", unexpected)

    if optimizer is not None:
        opt_path = directory / OPTIMIZER_PT
        if opt_path.exists():
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=map_location, weights_only=False)
            )
        else:
            _LOG.warning("no optimizer.pt in %s; optimizer state left untouched", directory)

    _LOG.info("Loaded checkpoint from %s (step=%d)", directory, meta.step)
    return meta


# ---------------------------------------------------------------------------
# R2 upload
# ---------------------------------------------------------------------------


def _split_uri(uri: str) -> tuple[str, str]:
    for scheme in ("r2://", "s3://"):
        if uri.startswith(scheme):
            rest = uri[len(scheme):]
            break
    else:
        raise ValueError(f"unknown upload URI scheme: {uri!r}")
    if "/" not in rest:
        raise ValueError(f"upload URI {uri!r} must contain '<bucket>/<prefix>'")
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def _upload_directory(directory: Path, uri: str) -> None:
    from another_world.data.storage_r2 import R2Client

    bucket, prefix = _split_uri(uri)
    r2 = R2Client.from_env()
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        key = f"{prefix}/{path.name}" if prefix else path.name
        with path.open("rb") as fh:
            r2.upload_fileobj(fh, bucket, key)
    _LOG.info("Uploaded checkpoint %s -> %s/%s", directory, bucket, prefix)


# ---------------------------------------------------------------------------
# Latest discovery
# ---------------------------------------------------------------------------


def find_latest_checkpoint(root: str | os.PathLike[str]) -> Path | None:
    """Return the checkpoint directory with the largest ``step`` under ``root``."""

    root = Path(root)
    if not root.exists():
        return None
    best: tuple[int, Path] | None = None
    for candidate in root.iterdir():
        meta_file = candidate / META_JSON
        if not meta_file.exists():
            continue
        try:
            meta = CheckpointMeta.from_json(meta_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if best is None or meta.step > best[0]:
            best = (meta.step, candidate)
    return None if best is None else best[1]


__all__ = [
    "CheckpointMeta",
    "find_latest_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]
