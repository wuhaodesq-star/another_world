# Architecture (draft)

This document expands on the architecture sketch in
[`roadmap.md`](roadmap.md). It will be filled in alongside the
implementation work in stages 1-4.

## 1. Modality tokenizers (stage 1-2)

| Modality | Default tokenizer | Output rate |
|----------|-------------------|-------------|
| Video    | Cosmos-Tokenizer CV8x8x8 | 1 token per (8t x 16x x 16y) cube |
| Image    | Cosmos-Tokenizer DI16x16 | 1 token per 16x16 patch |
| Text     | LLaMA-3 BPE              | ~3.5 chars per token |
| Action   | Learned per-env codebook | 1 token per environment step |

All tokens are mapped into a shared 100k vocabulary:
`[text 32k] + [visual 65k] + [action 4k] + [special 256]`.

## 2. Central dynamics model (stage 3)

- LLaMA-style decoder-only Transformer.
- Mixed positional encoding: RoPE-2D for visual tokens, RoPE-1D for text/action.
- Grouped-query attention, SwiGLU MLP, RMSNorm.
- FlashAttention-3 in production; CPU SDPA in tests.
- Three scale checkpoints: 7B, 30B (MoE), 70B (long-horizon).

## 3. Pixel decoder (stage 4)

- Conditional DiT (Diffusion Transformer) conditioned on token logits.
- Spatiotemporal joint attention (inspired by Open-Sora-Plan, Mochi).
- 3-stage curriculum: 256 -> 512 -> 1024.
- Inference acceleration: DPM-Solver, step distillation, TensorRT export.

## 4. Training objectives

```
L = alpha * NTP_multimodal
  + beta  * VJEPA_latent_prediction
  + gamma * action_conditioned_prediction
  + delta * rollout_consistency        (stage 6)
```

## 5. Inference pipeline (stage 4-8)

1. Encode reference image with Cosmos-Tokenizer.
2. Tokenize text prompt.
3. Autoregressive next-token prediction inside the dynamics model.
4. Detokenize visual tokens with the DiT decoder.
5. (Optional) Apply temporal post-processing.

Detailed module-level specs land in this document as each stage ships.
