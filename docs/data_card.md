# Data Card (draft)

This document inventories all data sources used to train Another World.
It is updated as each batch is approved and ingested.

## Sources (planned)

### Public datasets (stage 1.1)

| Name | Modality | Size | License | Status |
|------|----------|------|---------|--------|
| WebVid-10M | Video + caption | ~13 TB | Research-use | Planned |
| Panda-70M | Video + caption | ~100 TB | CC-style | Planned |
| HowTo100M | Video + ASR | ~30 TB | Research-use | Planned |
| Ego4D | Egocentric video | ~10 TB | Research-use | Planned |
| Kinetics-700 | Short clips + labels | ~1 TB | Research-use | Planned |
| LAION-2B | Image + caption | ~250 TB | CC-style | Planned |
| The Pile | Text | ~825 GB | Mixed open | Planned |

### Crawled (stage 1.2)

Each crawl batch requires explicit owner approval before kicking off (proxy /
target list / quota). Approved batches are appended below with manifest hashes.

| Batch | Source | Target count | Status | Manifest hash |
|-------|--------|--------------|--------|---------------|
| (none yet) | - | - | - | - |

## Filtering pipeline

1. Scene-level split (PySceneDetect).
2. Aesthetic predictor (LAION) >= threshold.
3. Watermark / NSFW filter.
4. Perceptual + URL dedup.
5. Whisper-large-v3 ASR.
6. Qwen2-VL automatic captioning.
7. Visual tokenisation with Cosmos-Tokenizer.

## Storage

Cloudflare R2 buckets:

- `another-world-raw`        : original media + manifests.
- `another-world-shards`     : WebDataset shards ready for training.
- `another-world-tokens`     : pre-tokenised shards (visual + text + action).

## Takedown / governance

- Open an issue or private security advisory in the repository.
- Removal target: < 7 days from confirmed report.
- Future training mixes exclude removed items at the manifest layer.
