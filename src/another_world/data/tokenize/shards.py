"""Token shard writer / reader.

A *token shard* is a tar archive containing pre-tokenised samples in the
standard WebDataset key/extension layout:

    000000.visual.pt
    000000.text.pt        # optional
    000000.action.pt      # optional
    000000.json           # metadata (key, source, license, fps, ...)
    000001.visual.pt
    ...

Each ``.pt`` file is a single tensor serialised with :func:`torch.save`.
The ``.json`` file holds free-form metadata. Files sharing the same numeric
prefix make up one sample.

We write **uncompressed tar** for fast random shard reads from object storage
(R2 supports HTTP range requests on plain tar). Compression is an opt-in.

Why not just pickle one big tensor per shard? Because WebDataset is the
default sample-streaming format used everywhere in our pipeline (and by
TorchTitan); keeping a single ``decode`` path for both raw video shards
and token shards minimises code duplication.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from another_world.data.datasets.sample import TokenSample
from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)

SHARD_SUFFIX = ".tar"
_VISUAL_KEY = "visual.pt"
_TEXT_KEY = "text.pt"
_ACTION_KEY = "action.pt"
_META_KEY = "json"


# ---------------------------------------------------------------------------
# Sample serialisation
# ---------------------------------------------------------------------------


def _save_tensor(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor.contiguous(), buf)
    return buf.getvalue()


def _load_tensor(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _sample_to_files(
    sample: TokenSample, *, sample_id: str
) -> list[tuple[str, bytes]]:
    """Return ``[(member_name, bytes), ...]`` for one sample."""

    out: list[tuple[str, bytes]] = []

    meta: dict[str, Any] = {
        "key": sample.key or sample_id,
        "extra": sample.extra,
        "visual_shape": list(sample.visual_tokens.shape),
        "visual_dtype": str(sample.visual_tokens.dtype),
    }
    if sample.text_tokens is not None:
        meta["text_shape"] = list(sample.text_tokens.shape)
        meta["text_dtype"] = str(sample.text_tokens.dtype)
    if sample.action_tokens is not None:
        meta["action_shape"] = list(sample.action_tokens.shape)
        meta["action_dtype"] = str(sample.action_tokens.dtype)

    out.append((f"{sample_id}.{_META_KEY}", json.dumps(meta).encode("utf-8")))
    out.append((f"{sample_id}.{_VISUAL_KEY}", _save_tensor(sample.visual_tokens)))
    if sample.text_tokens is not None:
        out.append((f"{sample_id}.{_TEXT_KEY}", _save_tensor(sample.text_tokens)))
    if sample.action_tokens is not None:
        out.append((f"{sample_id}.{_ACTION_KEY}", _save_tensor(sample.action_tokens)))
    return out


# ---------------------------------------------------------------------------
# Shard manifest (lightweight index per shard)
# ---------------------------------------------------------------------------


@dataclass
class ShardManifest:
    """Per-shard metadata written alongside the tar."""

    shard_path: str
    shard_id: str
    num_samples: int
    bytes_written: int
    created_at: float
    tokenizer: str | None = None
    source: str | None = None
    keys: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ShardManifest":
        return cls(**json.loads(text))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass
class TokenShardWriter:
    """Write a single token shard tar to disk.

    Designed to be used as a context manager so the tar is finalised even
    on exceptions.  When ``target_size_bytes`` is set, ``append`` returns
    ``True`` once the shard hit the cap, telling the caller to rotate to
    a new file.
    """

    path: str | os.PathLike[str]
    target_size_bytes: int | None = None
    compression: str | None = None  # None | "gz" | "bz2" | "xz"
    tokenizer: str | None = None
    source: str | None = None

    _tar: tarfile.TarFile | None = field(default=None, init=False, repr=False)
    _id_counter: int = field(default=0, init=False, repr=False)
    _bytes_written: int = field(default=0, init=False, repr=False)
    _keys: list[str] = field(default_factory=list, init=False, repr=False)
    _shard_id: str = field(default="", init=False, repr=False)
    _path: Path = field(default_factory=Path, init=False, repr=False)

    def __enter__(self) -> "TokenShardWriter":
        self._path = Path(self.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self.compression else f"w:{self.compression}"
        self._tar = tarfile.open(self._path, mode=mode)
        self._shard_id = self._path.stem
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def append(self, sample: TokenSample, *, sample_id: str | None = None) -> bool:
        if self._tar is None:
            raise RuntimeError("writer is not open; use it as a context manager")
        sid = sample_id or f"{self._id_counter:08d}"
        files = _sample_to_files(sample, sample_id=sid)
        now = time.time()
        for name, payload in files:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = int(now)
            info.mode = 0o644
            self._tar.addfile(info, io.BytesIO(payload))
            self._bytes_written += len(payload)
        self._keys.append(sample.key or sid)
        self._id_counter += 1

        if self.target_size_bytes is not None:
            return self._bytes_written >= self.target_size_bytes
        return False

    def close(self) -> ShardManifest | None:
        if self._tar is None:
            return None
        self._tar.close()
        self._tar = None
        manifest = ShardManifest(
            shard_path=str(self._path),
            shard_id=self._shard_id,
            num_samples=self._id_counter,
            bytes_written=self._bytes_written,
            created_at=time.time(),
            tokenizer=self.tokenizer,
            source=self.source,
            keys=list(self._keys),
        )
        manifest_path = self._path.with_suffix(".manifest.json")
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        _LOG.info(
            "Closed shard %s (%d samples, %.2f MB) + manifest %s",
            self._path, manifest.num_samples,
            manifest.bytes_written / (1024 * 1024), manifest_path,
        )
        return manifest

    @property
    def num_samples(self) -> int:
        return self._id_counter

    @property
    def bytes_written(self) -> int:
        return self._bytes_written


# ---------------------------------------------------------------------------
# Rotating writer
# ---------------------------------------------------------------------------


class RotatingShardWriter:
    """Spread samples across multiple shards based on a target size.

    Filenames follow ``{prefix}-{NNNNN}.tar``. The current shard is
    finalised (and a manifest written) automatically once it exceeds
    ``target_size_bytes``.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        *,
        prefix: str = "shard",
        target_size_bytes: int = 1024 * 1024 * 1024,  # 1 GB
        compression: str | None = None,
        tokenizer: str | None = None,
        source: str | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.target_size_bytes = target_size_bytes
        self.compression = compression
        self.tokenizer = tokenizer
        self.source = source

        self._shard_idx = 0
        self._current: TokenShardWriter | None = None
        self._manifests: list[ShardManifest] = []

    def _open_new(self) -> None:
        if self._current is not None:
            done = self._current.close()
            if done is not None:
                self._manifests.append(done)
        ext = ".tar" if not self.compression else f".tar.{self.compression}"
        path = self.out_dir / f"{self.prefix}-{self._shard_idx:05d}{ext}"
        self._current = TokenShardWriter(
            path=path,
            target_size_bytes=self.target_size_bytes,
            compression=self.compression,
            tokenizer=self.tokenizer,
            source=self.source,
        ).__enter__()
        self._shard_idx += 1

    def write(self, sample: TokenSample) -> None:
        if self._current is None:
            self._open_new()
        assert self._current is not None
        full = self._current.append(sample)
        if full:
            self._open_new()

    def close(self) -> list[ShardManifest]:
        if self._current is not None:
            done = self._current.close()
            if done is not None:
                self._manifests.append(done)
            self._current = None
        return list(self._manifests)

    def __enter__(self) -> "RotatingShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_token_shard(path: str | os.PathLike[str]) -> Iterator[TokenSample]:
    """Iterate :class:`TokenSample`s out of a single shard tar."""

    p = Path(path)
    mode = "r:*"  # auto-detect compression
    with tarfile.open(p, mode=mode) as tf:
        # Group members by sample id (the stem before the first dot).
        groups: dict[str, dict[str, bytes]] = {}
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            stem, _, ext_tail = name.partition(".")
            ext = ext_tail or ""
            payload = tf.extractfile(member)
            if payload is None:
                continue
            data = payload.read()
            groups.setdefault(stem, {})[ext] = data
        for sid in sorted(groups):
            files = groups[sid]
            try:
                yield _files_to_sample(sid, files)
            except Exception as exc:  # noqa: BLE001 - keep stream flowing
                _LOG.warning("Skipping bad sample %s in %s: %s", sid, p, exc)


def _files_to_sample(sample_id: str, files: dict[str, bytes]) -> TokenSample:
    if _VISUAL_KEY not in files:
        raise ValueError(f"sample {sample_id} missing {_VISUAL_KEY}")
    visual = _load_tensor(files[_VISUAL_KEY])
    text = _load_tensor(files[_TEXT_KEY]) if _TEXT_KEY in files else None
    action = (
        _load_tensor(files[_ACTION_KEY]) if _ACTION_KEY in files else None
    )
    meta: dict[str, Any] = {}
    if _META_KEY in files:
        try:
            meta = json.loads(files[_META_KEY].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta = {}
    return TokenSample(
        visual_tokens=visual,
        text_tokens=text,
        action_tokens=action,
        key=meta.get("key", sample_id),
        extra=meta.get("extra", {}),
    )


def read_token_shards(
    paths: Iterable[str | os.PathLike[str]],
) -> Iterator[TokenSample]:
    """Iterate samples across multiple shards in order."""

    for p in paths:
        yield from read_token_shard(p)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def make_sample_id() -> str:
    """Return a fresh sample id derived from a UUID4 (12 hex chars)."""

    return uuid.uuid4().hex[:12]


__all__ = [
    "RotatingShardWriter",
    "SHARD_SUFFIX",
    "ShardManifest",
    "TokenShardWriter",
    "make_sample_id",
    "read_token_shard",
    "read_token_shards",
]
