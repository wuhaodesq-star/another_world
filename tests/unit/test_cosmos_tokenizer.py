"""Tests for the Cosmos-Tokenizer wrapper.

We mock the underlying ``CausalVideoTokenizer`` so these tests can run on
CPU without the proprietary HuggingFace weights or the ``cosmos_tokenizer``
PyPI package.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from another_world.tokenizers.visual import cosmos


def _make_fake_ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "encoder.jit").write_bytes(b"")
    (d / "decoder.jit").write_bytes(b"")
    (d / "autoencoder.jit").write_bytes(b"")
    return d


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_video_model_is_registered() -> None:
    assert cosmos.DEFAULT_VIDEO_MODEL in cosmos.COSMOS_REGISTRY


def test_list_models_returns_sorted() -> None:
    names = cosmos.list_models()
    assert names == sorted(names)
    assert all(name.startswith("Cosmos-") for name in names)


def test_get_spec_round_trip() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    assert spec.kind == "discrete"
    assert spec.spatial_compression == 16
    assert spec.temporal_compression == 8
    assert spec.vocab_size == 64000
    assert spec.hf_repo == "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"


def test_get_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        cosmos.get_spec("not-a-model")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_video_input_accepts_canonical_shape() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    v = torch.zeros(1, 3, 17, 256, 256)
    cosmos.validate_video_input(v, spec)  # should not raise


def test_validate_video_input_rejects_bad_frames() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    v = torch.zeros(1, 3, 10, 256, 256)  # 10 != 1 + k*8
    with pytest.raises(ValueError, match="temporal_compression"):
        cosmos.validate_video_input(v, spec)


def test_validate_video_input_rejects_bad_spatial() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    v = torch.zeros(1, 3, 9, 250, 256)  # 250 % 16 != 0
    with pytest.raises(ValueError, match="spatial"):
        cosmos.validate_video_input(v, spec)


def test_validate_video_input_rejects_wrong_dims() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    with pytest.raises(ValueError, match=r"\[B, C, T, H, W\]"):
        cosmos.validate_video_input(torch.zeros(3, 9, 256, 256), spec)


def test_validate_video_input_rejects_wrong_channels() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    with pytest.raises(ValueError, match="color channels"):
        cosmos.validate_video_input(torch.zeros(1, 4, 9, 256, 256), spec)


def test_validate_video_input_rejects_non_tensor() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    with pytest.raises(TypeError):
        cosmos.validate_video_input([1, 2, 3], spec)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Latent shape inference
# ---------------------------------------------------------------------------


def test_latent_shape_discrete() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-DV8x16x16")
    shape = cosmos.expected_latent_shape(spec, batch=2, frames=17, height=256, width=256)
    # T' = 1 + (17-1)/8 = 3,  H'=W'=256/16=16
    assert shape == (2, 3, 16, 16)


def test_latent_shape_continuous() -> None:
    spec = cosmos.get_spec("Cosmos-1.0-Tokenizer-CV8x8x8")
    shape = cosmos.expected_latent_shape(spec, batch=1, frames=9, height=512, width=512)
    # C=16, T'=1+(9-1)/8=2, H'=W'=512/8=64
    assert shape == (1, 16, 2, 64, 64)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def test_checkpoints_from_directory(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    ck = cosmos.CosmosCheckpoints.from_directory(d)
    assert ck.encoder == d / "encoder.jit"
    assert ck.decoder == d / "decoder.jit"
    assert ck.autoencoder == d / "autoencoder.jit"


def test_checkpoints_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cosmos.CosmosCheckpoints.from_directory(tmp_path / "nope")


def test_checkpoints_partial_files(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "encoder.jit").write_bytes(b"")
    ck = cosmos.CosmosCheckpoints.from_directory(d)
    assert ck.encoder is not None
    assert ck.decoder is None
    with pytest.raises(FileNotFoundError):
        ck.require_decoder()


# ---------------------------------------------------------------------------
# Wrapper construction (with mocked native tokenizer)
# ---------------------------------------------------------------------------


def test_wrapper_from_local_rejects_image_models(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    with pytest.raises(ValueError, match="image"):
        cosmos.CosmosVideoTokenizer.from_local(
            "Cosmos-0.1-Tokenizer-DI8x8", d, device="cpu"
        )


def test_wrapper_from_local_builds(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-DV8x16x16", d, device="cpu", dtype=torch.float32
    )
    assert tk.spec.kind == "discrete"
    assert tk.device.type == "cpu"


def test_encode_validates_before_calling_native(tmp_path: Path) -> None:
    """If validation fails, the native model must never be touched."""
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-DV8x16x16", d, device="cpu", dtype=torch.float32
    )
    # No encoder built yet -> a successful call would try to import the real
    # library. We deliberately pass a bad shape so we exit early.
    bad = torch.zeros(1, 3, 4, 256, 256)
    with pytest.raises(ValueError):
        tk.encode(bad)
    assert tk._encoder is None  # noqa: SLF001


def test_encode_decode_with_injected_mock(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-DV8x16x16", d, device="cpu", dtype=torch.float32
    )
    enc = MagicMock()
    dec = MagicMock()
    fake_indices = torch.zeros(1, 3, 16, 16, dtype=torch.long)
    fake_codes = torch.zeros(1, 6, 3, 16, 16)
    fake_recon = torch.zeros(1, 3, 17, 256, 256)
    enc.encode.return_value = (fake_indices, fake_codes)
    dec.decode.return_value = fake_recon
    tk._encoder = enc  # noqa: SLF001
    tk._decoder = dec  # noqa: SLF001

    v = torch.zeros(1, 3, 17, 256, 256)
    indices, codes = tk.encode(v)
    assert indices.shape == (1, 3, 16, 16)
    assert codes.shape == (1, 6, 3, 16, 16)
    enc.encode.assert_called_once()

    recon = tk.decode(indices)
    assert recon.shape == v.shape
    dec.decode.assert_called_once()


def test_encode_single_tensor_return_is_wrapped(tmp_path: Path) -> None:
    """Continuous tokenizers may return a single tensor; we wrap into a tuple."""
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-CV8x8x8", d, device="cpu", dtype=torch.float32
    )
    enc = MagicMock()
    enc.encode.return_value = torch.zeros(1, 16, 2, 64, 64)  # not a tuple
    tk._encoder = enc  # noqa: SLF001
    v = torch.zeros(1, 3, 9, 512, 512)
    out = tk.encode(v)
    assert isinstance(out, tuple)
    assert out[0].shape == (1, 16, 2, 64, 64)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_round_frames(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-DV8x16x16", d, device="cpu", dtype=torch.float32
    )
    assert tk.round_frames(1) == 1
    assert tk.round_frames(2) == 9
    assert tk.round_frames(9) == 9
    assert tk.round_frames(10) == 17
    assert tk.round_frames(17) == 17
    with pytest.raises(ValueError):
        tk.round_frames(0)


def test_latent_shape_for_helper(tmp_path: Path) -> None:
    d = _make_fake_ckpt_dir(tmp_path)
    tk = cosmos.CosmosVideoTokenizer.from_local(
        "Cosmos-1.0-Tokenizer-DV8x16x16", d, device="cpu", dtype=torch.float32
    )
    assert tk.latent_shape_for(batch=2, frames=17, height=256, width=256) == (
        2, 3, 16, 16,
    )
