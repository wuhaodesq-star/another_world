#!/usr/bin/env python
"""Crawl manifest CLI.

Workflow:

1. Propose a batch (creates JSON, status=proposed)::

       python scripts/crawl_manifest.py propose \
           --batch-id ytcc-2026-w01 --source youtube \
           --urls-file targets.txt --owner wuhaodesq-star \
           --max-count 100 --min-res 720 1280

2. Owner reviews ``manifests/ytcc-2026-w01.json`` and replies "approved" in chat.

3. Operator marks it approved::

       python scripts/crawl_manifest.py approve \
           --path manifests/ytcc-2026-w01.json --approver wuhaodesq-star \
           --comment "ok 2026-06-25"

4. Crawlers refuse to start unless the manifest is freshly approved
   (enforced by ``another_world.data.crawlers.gate_crawl``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from another_world.data.crawlers import (
    CrawlConstraints,
    CrawlManifest,
    CrawlTarget,
    gate_crawl,
)


def _read_urls(path: Path) -> list[CrawlTarget]:
    out: list[CrawlTarget] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        out.append(
            CrawlTarget(
                url=parts[0],
                license=parts[1] if len(parts) > 1 else None,
                title=parts[2] if len(parts) > 2 else None,
            )
        )
    return out


def _cmd_propose(args: argparse.Namespace) -> int:
    targets = _read_urls(Path(args.urls_file))
    constraints = CrawlConstraints(
        max_count=args.max_count,
        max_total_seconds=args.max_total_seconds,
        min_resolution=tuple(args.min_res) if args.min_res else None,
        allowed_licenses=tuple(args.allowed_licenses),
    )
    manifest = CrawlManifest(
        batch_id=args.batch_id,
        source=args.source,
        targets=targets,
        constraints=constraints,
        proposed_by=args.owner,
        notes=args.notes,
    )
    out = Path(args.out) if args.out else Path("manifests") / f"{args.batch_id}.json"
    manifest.save(out)
    print(manifest.summary())
    print(f"\nManifest saved to {out}")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    manifest = CrawlManifest.load(args.path)
    manifest.approve(approver=args.approver, comment=args.comment)
    manifest.save(args.path)
    print(manifest.summary())
    print(f"\nManifest at {args.path} is now APPROVED.")
    return 0


def _cmd_revoke(args: argparse.Namespace) -> int:
    manifest = CrawlManifest.load(args.path)
    manifest.revoke(reason=args.reason)
    manifest.save(args.path)
    print(manifest.summary())
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    manifest = CrawlManifest.load(args.path)
    print(manifest.summary())
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    manifest = CrawlManifest.load(args.path)
    try:
        gate_crawl(manifest)
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print("OK to crawl.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("crawl_manifest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_propose = sub.add_parser("propose", help="create a new manifest")
    p_propose.add_argument("--batch-id", required=True)
    p_propose.add_argument("--source", required=True,
                           choices=["youtube", "vimeo", "commons", "local"])
    p_propose.add_argument("--urls-file", required=True, type=Path)
    p_propose.add_argument("--owner", default=None)
    p_propose.add_argument("--out", default=None)
    p_propose.add_argument("--max-count", type=int, default=None)
    p_propose.add_argument("--max-total-seconds", type=float, default=None)
    p_propose.add_argument("--min-res", type=int, nargs=2, default=None,
                           metavar=("HEIGHT", "WIDTH"))
    p_propose.add_argument("--allowed-licenses", nargs="+",
                           default=["cc-by", "cc-by-sa", "cc0", "public-domain"])
    p_propose.add_argument("--notes", default=None)
    p_propose.set_defaults(func=_cmd_propose)

    p_approve = sub.add_parser("approve", help="mark a manifest approved")
    p_approve.add_argument("--path", required=True, type=Path)
    p_approve.add_argument("--approver", required=True)
    p_approve.add_argument("--comment", default=None)
    p_approve.set_defaults(func=_cmd_approve)

    p_revoke = sub.add_parser("revoke", help="revoke a manifest")
    p_revoke.add_argument("--path", required=True, type=Path)
    p_revoke.add_argument("--reason", required=True)
    p_revoke.set_defaults(func=_cmd_revoke)

    p_inspect = sub.add_parser("inspect", help="print manifest summary")
    p_inspect.add_argument("--path", required=True, type=Path)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_gate = sub.add_parser("gate", help="check if a manifest is OK to crawl")
    p_gate.add_argument("--path", required=True, type=Path)
    p_gate.set_defaults(func=_cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
