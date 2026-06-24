#!/usr/bin/env python
"""Stub for the dataset preparation pipeline (stage 1).

Will orchestrate:

1. yt-dlp crawl with proxy + CC license filter.
2. PySceneDetect splitting + ffmpeg re-encode.
3. ASR (Whisper-large-v3) and caption (Qwen2-VL) generation.
4. Aesthetic / NSFW / dedup filtering.
5. Visual + text tokenization with the Cosmos-Tokenizer.
6. Pack into WebDataset shards and upload to Cloudflare R2.

For now this script only prints the planned pipeline so we have a single
entry point we can fill in stage 1.2.
"""

from __future__ import annotations

import argparse


STAGES = [
    "crawl",
    "split",
    "asr",
    "caption",
    "filter",
    "tokenize",
    "pack",
    "upload",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a multimodal data shard.")
    parser.add_argument("--source", choices=["youtube", "vimeo", "local"], required=False)
    parser.add_argument("--out-bucket", default="another-world-data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("[prepare_data] planned pipeline:")
    for i, stage in enumerate(STAGES, start=1):
        print(f"  {i}. {stage}")
    print(f"[prepare_data] source={args.source!r} bucket={args.out_bucket!r} "
          f"dry_run={args.dry_run}")
    print("[prepare_data] implementation lands in stage 1.2 (after explicit "
          "owner approval per crawl batch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
