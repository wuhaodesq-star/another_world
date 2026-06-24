# Model Card (draft)

This document will describe each released checkpoint. As of stage 0 no
checkpoints exist; this card is a template that the release process fills in.

## another-world-7b-v0.1 (planned)

| Field | Value |
|-------|-------|
| Status | Planned (target: month 6) |
| Architecture | Decoder-only Transformer (7B params, GQA, SwiGLU, RoPE) |
| Visual tokenizer | Cosmos-Tokenizer CV8x8x8 (frozen) |
| Pixel decoder | DiT, 3B params, 3-stage curriculum |
| Training tokens | ~1 T multimodal |
| Hardware | 256 x H100 (Lambda Labs), bf16 + FSDP2 |
| Context length | 8 k tokens (~30 s of video) |
| Inputs | 1 reference image + text prompt (<= 77 tokens) |
| Outputs | 5 s video, 512 x 288, 24 fps |
| License | Apache-2.0 (weights TBD) |

### Intended use

- Research on multimodal world models.
- Foundation for downstream RL / robotics simulators.
- Educational demonstrations.

### Out-of-scope use

- Real-time safety-critical control.
- Generation of misleading media without consent.
- Surveillance applications.

### Evaluation results

Filled in at release time.
