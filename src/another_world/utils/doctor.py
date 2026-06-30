"""``aw-doctor`` system / environment diagnostics CLI.

Quick health check that prints whether each major optional dependency
is installed and what the runtime sees. Useful right after spinning up
a fresh Lambda Labs box or in CI to confirm the environment is set up
correctly before launching a long training job.

Each check is a pure function returning a (name, ok, detail) tuple so
they can be reused programmatically.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    return CheckResult(
        "python>=3.10",
        ok,
        f"{version} ({sys.executable})",
    )


def _check_torch() -> CheckResult:
    try:
        import torch
    except ImportError as exc:
        return CheckResult("torch", False, f"not installed ({exc})")
    cuda = "cuda" if torch.cuda.is_available() else "cpu-only"
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return CheckResult(
        "torch",
        True,
        f"{torch.__version__} ({cuda}, {n_gpu} GPU(s))",
    )


def _check_optional_module(name: str) -> Callable[[], CheckResult]:
    def _inner() -> CheckResult:
        try:
            mod = __import__(name)
        except ImportError as exc:
            return CheckResult(name, False, f"not installed ({exc})")
        version = getattr(mod, "__version__", "?")
        return CheckResult(name, True, version)

    _inner.__name__ = f"_check_{name}"
    return _inner


def _check_env(name: str) -> Callable[[], CheckResult]:
    def _inner() -> CheckResult:
        val = os.environ.get(name)
        return CheckResult(
            f"env:{name}",
            val is not None,
            "set" if val is not None else "missing",
        )

    _inner.__name__ = f"_check_env_{name}"
    return _inner


def _check_aw_imports() -> CheckResult:
    try:
        import another_world  # noqa: F401
        from another_world.models.dynamics import MultimodalDynamicsModel  # noqa: F401
        from another_world.models.decoder import DiTDecoder  # noqa: F401
        from another_world.inference.generation import generate  # noqa: F401
        from another_world.training.multimodal import run_multimodal_training  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return CheckResult("another_world imports", False, str(exc))
    return CheckResult("another_world imports", True, "all core modules importable")


def _check_platform() -> CheckResult:
    return CheckResult(
        "platform",
        True,
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def run_checks(*, include_optional: bool = True) -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        _check_python,
        _check_platform,
        _check_torch,
        _check_aw_imports,
    ]
    if include_optional:
        checks += [
            _check_optional_module("numpy"),
            _check_optional_module("transformers"),
            _check_optional_module("safetensors"),
            _check_optional_module("webdataset"),
            _check_optional_module("av"),
            _check_optional_module("boto3"),
            _check_optional_module("wandb"),
            _check_optional_module("gradio"),
            _check_optional_module("PIL"),
            _check_optional_module("huggingface_hub"),
        ]
        checks += [
            _check_env("WANDB_API_KEY"),
            _check_env("HF_TOKEN"),
            _check_env("R2_ACCOUNT_ID"),
            _check_env("R2_ACCESS_KEY_ID"),
        ]
    return [c() for c in checks]


def render_text(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines: list[str] = []
    for r in results:
        mark = "OK " if r.ok else "MISS"
        lines.append(f"[{mark}] {r.name.ljust(width)}  {r.detail}")
    ok_count = sum(1 for r in results if r.ok)
    lines.append("")
    lines.append(f"{ok_count}/{len(results)} checks passed")
    return "\n".join(lines)


def render_json(results: list[CheckResult]) -> str:
    import json

    payload = [
        {"name": r.name, "ok": r.ok, "detail": r.detail} for r in results
    ]
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aw-doctor",
        description="Print environment diagnostics for the Another World project.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="only run core checks (Python, torch, package imports)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any check failed (including optional ones)",
    )
    args = parser.parse_args(argv)

    results = run_checks(include_optional=not args.no_optional)
    output = render_json(results) if args.json else render_text(results)
    print(output)

    if args.strict and any(not r.ok for r in results):
        return 1
    # Always fail when *core* checks fail, even without --strict.
    core_names = {"python>=3.10", "torch", "another_world imports"}
    if any(not r.ok for r in results if r.name in core_names):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
