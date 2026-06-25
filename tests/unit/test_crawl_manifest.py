"""Tests for the crawl manifest data model and approval workflow."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from another_world.data.crawlers import (
    CrawlConstraints,
    CrawlManifest,
    CrawlTarget,
    gate_crawl,
)
from another_world.data.crawlers.manifest import MAX_APPROVAL_AGE_SECONDS


def _make_manifest() -> CrawlManifest:
    return CrawlManifest(
        batch_id="batch001",
        source="youtube",
        targets=[
            CrawlTarget(url="https://example.com/a", license="CC-BY"),
            CrawlTarget(url="https://example.com/b", license="CC-BY-SA"),
        ],
        constraints=CrawlConstraints(
            max_count=10, min_resolution=(720, 1280),
        ),
        proposed_by="wuhaodesq-star",
    )


def test_default_status_is_proposed() -> None:
    m = _make_manifest()
    assert m.status == "proposed"
    assert m.approval is None


def test_approve_changes_status_and_stamps_approver() -> None:
    m = _make_manifest()
    m.approve("wuhaodesq-star", comment="ok")
    assert m.status == "approved"
    assert m.approval is not None
    assert m.approval.approver == "wuhaodesq-star"
    assert m.approval.comment == "ok"


def test_approve_rejects_invalid_status() -> None:
    m = _make_manifest()
    m.approve("owner")
    m.mark_in_progress()
    with pytest.raises(RuntimeError):
        m.approve("owner")


def test_revoke_marks_status() -> None:
    m = _make_manifest()
    m.approve("owner")
    m.revoke("license review failed")
    assert m.status == "revoked"
    assert "license review failed" in (m.notes or "")


def test_gate_refuses_unapproved() -> None:
    m = _make_manifest()
    with pytest.raises(RuntimeError, match="approved"):
        gate_crawl(m)


def test_gate_refuses_stale_approval() -> None:
    m = _make_manifest()
    m.approve("owner")
    # Backdate approval beyond the max age.
    m.approval.approved_at = time.time() - MAX_APPROVAL_AGE_SECONDS - 1  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="re-approve"):
        gate_crawl(m)


def test_mark_in_progress_requires_fresh_approval() -> None:
    m = _make_manifest()
    with pytest.raises(RuntimeError):
        m.mark_in_progress()
    m.approve("owner")
    m.mark_in_progress()
    assert m.status == "in_progress"


def test_round_trip_save_and_load(tmp_path: Path) -> None:
    m = _make_manifest()
    m.approve("owner", comment="batch ok")
    out = tmp_path / "manifests" / "batch001.json"
    m.save(out)
    reloaded = CrawlManifest.load(out)
    assert reloaded.batch_id == m.batch_id
    assert reloaded.status == "approved"
    assert reloaded.approval is not None
    assert reloaded.approval.approver == "owner"
    assert reloaded.targets[0].url == "https://example.com/a"
    assert reloaded.constraints.min_resolution == (720, 1280)
    assert reloaded.constraints.allowed_licenses == m.constraints.allowed_licenses


def test_summary_contains_key_fields() -> None:
    m = _make_manifest()
    text = m.summary()
    assert "batch001" in text
    assert "youtube" in text
    assert "Targets" in text


def test_from_dict_handles_missing_optional_fields() -> None:
    data = {
        "batch_id": "x",
        "source": "local",
        "targets": [{"url": "u"}],
    }
    m = CrawlManifest.from_dict(data)
    assert m.batch_id == "x"
    assert len(m.targets) == 1
    assert m.status == "proposed"
