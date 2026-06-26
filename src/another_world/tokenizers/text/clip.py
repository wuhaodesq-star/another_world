"""CLIP text encoder wrapper.

Unlike :mod:`encoders`, which returns token ids for the dynamics model,
this module returns continuous CLIP text embeddings useful for:

- text/video alignment metrics,
- conditioning a DiT or reward model,
- prompt embedding caches.

The implementation is a lazy wrapper around HuggingFace ``transformers``
(``CLIPTokenizer`` + ``CLIPTextModel``). It has no effect on CI unless
instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class CLIPTextEncoder:
    """Lazy HuggingFace CLIP text encoder.

    Args:
        model_name: HF model id (default: OpenAI CLIP ViT-B/32).
        device: ``auto`` / ``cpu`` / ``cuda``.
        normalize: whether to L2-normalize output embeddings.
    """

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    normalize: bool = True
    max_length: int = 77

    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    def _device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _ensure(self) -> tuple[object, object]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for CLIPTextEncoder "
                "(`pip install transformers`)."
            ) from exc
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self._model = CLIPTextModel.from_pretrained(self.model_name).to(self._device()).eval()
        return self._tokenizer, self._model

    @torch.no_grad()
    def encode(self, texts: str | list[str]) -> Tensor:
        """Return pooled CLIP embeddings ``[B, D]``."""

        if isinstance(texts, str):
            texts = [texts]
        tokenizer, model = self._ensure()
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        out = model(**inputs)
        emb = out.pooler_output.float()
        if self.normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu()

    @torch.no_grad()
    def similarity(self, a: str | list[str], b: str | list[str]) -> Tensor:
        """Cosine similarity matrix between two text sets."""

        ea = self.encode(a)
        eb = self.encode(b)
        return ea @ eb.t()


__all__ = ["CLIPTextEncoder"]
