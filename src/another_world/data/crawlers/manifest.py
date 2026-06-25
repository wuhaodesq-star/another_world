"""Crawl manifest format and approval workflow.

The repository-wide rule is that **no crawl batch starts without explicit
owner approval**. This module implements the data structures that record
those approvals and the items in each batch.

A manifest is a JSON file (one per batch) that captures:

- ``batch_id``: opaque identifier (slug or hash).
- ``source``: ``youtube``, ``vimeo``, ``commons``, ``local``.
- ``proposed_at`` / ``approved_at`` timestamps.
- ``approver``: GitHub username of the project owner.
- ``targets``: a list of :class:`CrawlTarget`s (URL + expected license).
- ``constraints``: per-batch limits (max_count, max_duration, fps_range, ...).
- ``status``: ``proposed`` -> ``approved`` -> ``in_progress`` -> ``done`` /
  ``failed`` / ``revoked``.

Workflow
--------

1. A planner creates a :class:`CrawlManifest` with ``status="proposed"``
   and ``proposed_at`` set.
2. The script prints a one-page summary to stdout. The operator forwards
   it to the owner in chat.
3. The owner replies "approved" (or rejects); the operator runs
   :meth:`CrawlManifest.approve` with the owner's username.
4. The crawler refuses to start unless ``status == "approved"`` and the
   approval timestamp is no older than ``MAX_APPROVAL_AGE``.

This keeps a clean audit trail in the repository even when the actual
crawl runs on a different machine.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_APPROVAL_AGE_SECONDS = 7 * 24 * 3600  # one week


@dataclass
class CrawlTarget:
    """A single item we plan to fetch."""

    url: str
    license: str | None = None
    title: str | None = None
    duration_seconds: float | None = None
    notes: str | None = None


@dataclass
class CrawlConstraints:
    """Per-batch hard limits enforced before and during crawling."""

    max_count: int | None = None
    max_total_seconds: float | None = None
    min_resolution: tuple[int, int] | None = None
    allowed_licenses: tuple[str, ...] = (
        "cc-by",
        "cc-by-sa",
        "cc0",
        "public-domain",
    )
    safe_search: bool = True


@dataclass
class CrawlApproval:
    """Owner-supplied approval payload."""

    approver: str
    approved_at: float
    comment: str | None = None


@dataclass
class CrawlManifest:
    """Persistent record of a single crawl batch."""

    batch_id: str
    source: str
    targets: list[CrawlTarget]
    constraints: CrawlConstraints = field(default_factory=CrawlConstraints)
    status: str = "proposed"
    proposed_at: float = field(default_factory=time.time)
    proposed_by: str | None = None
    approval: CrawlApproval | None = None
    notes: str | None = None

    # ----- lifecycle -------------------------------------------------------

    def approve(self, approver: str, *, comment: str | None = None) -> None:
        if self.status not in ("proposed", "approved"):
            raise RuntimeError(
                f"cannot approve manifest in status '{self.status}'"
            )
        self.approval = CrawlApproval(
            approver=approver, approved_at=time.time(), comment=comment
        )
        self.status = "approved"

    def revoke(self, reason: str) -> None:
        self.status = "revoked"
        self.notes = (self.notes or "") + f"\nREVOKED: {reason}"

    def mark_in_progress(self) -> None:
        self._require_fresh_approval()
        self.status = "in_progress"

    def mark_done(self) -> None:
        self.status = "done"

    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.notes = (self.notes or "") + f"\nFAILED: {reason}"

    def _require_fresh_approval(self) -> None:
        if self.status != "approved":
            raise RuntimeError(
                f"manifest must be approved (got '{self.status}')"
            )
        if self.approval is None:
            raise RuntimeError("approval payload missing")
        age = time.time() - self.approval.approved_at
        if age > MAX_APPROVAL_AGE_SECONDS:
            raise RuntimeError(
                f"approval is {age / 3600:.1f}h old; re-approve before crawling"
            )

    # ----- I/O -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _convert(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CrawlManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlManifest":
        constraints_data = data.get("constraints") or {}
        constraints = CrawlConstraints(
            max_count=constraints_data.get("max_count"),
            max_total_seconds=constraints_data.get("max_total_seconds"),
            min_resolution=(
                tuple(constraints_data["min_resolution"])
                if constraints_data.get("min_resolution")
                else None
            ),
            allowed_licenses=tuple(
                constraints_data.get(
                    "allowed_licenses", CrawlConstraints().allowed_licenses
                )
            ),
            safe_search=constraints_data.get("safe_search", True),
        )
        approval_data = data.get("approval")
        approval = (
            CrawlApproval(
                approver=approval_data["approver"],
                approved_at=float(approval_data["approved_at"]),
                comment=approval_data.get("comment"),
            )
            if approval_data
            else None
        )
        targets = [
            CrawlTarget(
                url=t["url"],
                license=t.get("license"),
                title=t.get("title"),
                duration_seconds=t.get("duration_seconds"),
                notes=t.get("notes"),
            )
            for t in data.get("targets", [])
        ]
        return cls(
            batch_id=data["batch_id"],
            source=data["source"],
            targets=targets,
            constraints=constraints,
            status=data.get("status", "proposed"),
            proposed_at=float(data.get("proposed_at", time.time())),
            proposed_by=data.get("proposed_by"),
            approval=approval,
            notes=data.get("notes"),
        )

    # ----- presentation ----------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Batch     : {self.batch_id}",
            f"Source    : {self.source}",
            f"Status    : {self.status}",
            f"Proposed  : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.proposed_at))}",
            f"Targets   : {len(self.targets)}",
            f"Limits    : max_count={self.constraints.max_count}, "
            f"max_secs={self.constraints.max_total_seconds}, "
            f"min_res={self.constraints.min_resolution}, "
            f"licenses={list(self.constraints.allowed_licenses)}",
        ]
        if self.approval:
            lines.append(
                f"Approved  : {self.approval.approver} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.approval.approved_at))}"
            )
        if self.notes:
            lines.append(f"Notes     : {self.notes}")
        return "\n".join(lines)


def _convert(obj: Any) -> Any:
    """Recursively convert dataclass dicts so tuples become lists for JSON."""

    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def gate_crawl(manifest: CrawlManifest) -> None:
    """Raise unless ``manifest`` is freshly approved.

    Crawlers should call this *immediately before* starting any I/O so the
    process refuses to run without owner sign-off.
    """

    manifest._require_fresh_approval()  # noqa: SLF001


__all__ = [
    "CrawlApproval",
    "CrawlConstraints",
    "CrawlManifest",
    "CrawlTarget",
    "MAX_APPROVAL_AGE_SECONDS",
    "gate_crawl",
]
