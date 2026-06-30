"""Tests for the aw-doctor diagnostics CLI."""

from __future__ import annotations

import json

from another_world.utils.doctor import (
    CheckResult,
    main,
    render_json,
    render_text,
    run_checks,
)


def test_run_checks_core_only_passes() -> None:
    results = run_checks(include_optional=False)
    names = {r.name for r in results}
    assert "python>=3.10" in names
    assert "torch" in names
    assert "another_world imports" in names
    # core checks must pass in a properly configured dev env
    core = [r for r in results if r.name in
            {"python>=3.10", "torch", "another_world imports"}]
    assert all(r.ok for r in core), [r for r in core if not r.ok]


def test_run_checks_with_optional_includes_module_checks() -> None:
    results = run_checks(include_optional=True)
    names = {r.name for r in results}
    assert "numpy" in names
    assert "env:WANDB_API_KEY" in names
    assert "PIL" in names


def test_render_text_marks_ok_and_miss() -> None:
    results = [
        CheckResult("a", True, "ok"),
        CheckResult("b", False, "missing"),
    ]
    text = render_text(results)
    assert "[OK ]" in text
    assert "[MISS]" in text
    assert "1/2 checks passed" in text


def test_render_json_round_trips() -> None:
    results = [CheckResult("x", True, "1.2.3")]
    out = json.loads(render_json(results))
    assert out == [{"name": "x", "ok": True, "detail": "1.2.3"}]


def test_main_no_optional_returns_zero(capsys) -> None:
    rc = main(["--no-optional"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "python" in out


def test_main_strict_returns_nonzero_when_optional_missing(
    monkeypatch, capsys,
) -> None:
    # Force an optional env var to be unset so the strict run fails.
    for k in ("WANDB_API_KEY", "HF_TOKEN", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    rc = main(["--strict"])
    capsys.readouterr()
    assert rc == 1


def test_main_emits_json(capsys) -> None:
    rc = main(["--no-optional", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert {"name", "ok", "detail"}.issubset(parsed[0].keys())
