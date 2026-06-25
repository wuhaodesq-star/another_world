"""Tests for the filter primitives and pipeline composer."""

from __future__ import annotations

import torch

from another_world.data.datasets.sample import VideoSample
from another_world.data.filters import (
    AestheticFilter,
    AspectRatioFilter,
    CallableFilter,
    DedupFilter,
    FilterPipeline,
    LicenseFilter,
    MinDurationFilter,
    MinResolutionFilter,
)


def _sample(h: int = 256, w: int = 256, fps: float = 30.0,
            num_frames: int = 60, license: str | None = "CC-BY",
            value: int | None = None) -> VideoSample:
    frames = (
        torch.full((num_frames, 3, h, w), value, dtype=torch.uint8)
        if value is not None
        else torch.randint(0, 256, (num_frames, 3, h, w), dtype=torch.uint8)
    )
    return VideoSample(frames=frames, fps=fps, license=license)


def test_min_duration_keeps_long_clip() -> None:
    s = _sample(num_frames=120, fps=30.0)  # 4 seconds
    assert MinDurationFilter(min_seconds=2.0)(s) is s


def test_min_duration_drops_short_clip() -> None:
    s = _sample(num_frames=10, fps=30.0)  # ~0.33s
    assert MinDurationFilter(min_seconds=2.0)(s) is None


def test_min_duration_uses_duration_field_if_present() -> None:
    s = _sample(num_frames=2, fps=None)
    s.duration = 5.0
    assert MinDurationFilter(min_seconds=2.0)(s) is s


def test_min_duration_passes_when_undecidable() -> None:
    s = _sample(num_frames=2, fps=None)
    s.duration = None
    assert MinDurationFilter()(s) is s


def test_min_resolution() -> None:
    big = _sample(h=512, w=512)
    small = _sample(h=64, w=64)
    f = MinResolutionFilter(min_height=128, min_width=128)
    assert f(big) is big
    assert f(small) is None


def test_aspect_ratio_keeps_normal() -> None:
    s = _sample(h=240, w=320)
    assert AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)(s) is s


def test_aspect_ratio_drops_extreme() -> None:
    tall = _sample(h=480, w=128)  # 0.27 -> rejected
    flat = _sample(h=64, w=512)  # 8.0 -> rejected
    f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.5)
    assert f(tall) is None
    assert f(flat) is None


def test_license_filter_allow_list() -> None:
    cc = _sample(license="CC-BY-SA-4.0")
    rights = _sample(license="All Rights Reserved")
    f = LicenseFilter()
    assert f(cc) is cc
    assert f(rights) is None


def test_license_filter_missing_field() -> None:
    s = _sample(license=None)
    assert LicenseFilter(require_field=True)(s) is None
    assert LicenseFilter(require_field=False)(s) is s


def test_aesthetic_filter_drops_constant_frames() -> None:
    blank = _sample(value=0)
    assert AestheticFilter(min_variance=10.0)(blank) is None


def test_aesthetic_filter_records_variance() -> None:
    s = _sample()
    out = AestheticFilter(min_variance=0.0)(s)
    assert out is s
    assert "aesthetic_var" in s.extra["metrics"]


def test_dedup_filter_rejects_repeats() -> None:
    torch.manual_seed(0)
    s = _sample()
    f = DedupFilter()
    assert f(s) is s
    # Same content -> identical fingerprint -> dropped
    s2 = VideoSample(frames=s.frames.clone(), fps=s.fps, license=s.license)
    assert f(s2) is None


def test_dedup_fingerprint_is_stable() -> None:
    s = _sample()
    a = DedupFilter.fingerprint(s.frames)
    b = DedupFilter.fingerprint(s.frames.clone())
    assert a == b
    assert len(a) == 16


def test_callable_filter_wraps_plain_function() -> None:
    def drop_short(s: VideoSample) -> VideoSample | None:
        return None if s.num_frames < 5 else s

    f = CallableFilter("drop_short", drop_short)
    assert f(_sample(num_frames=10)) is not None
    assert f(_sample(num_frames=2)) is None


def test_pipeline_short_circuits_and_counts() -> None:
    samples = [
        _sample(num_frames=100, fps=30, license="CC-BY"),
        _sample(num_frames=2, fps=30, license="CC-BY"),
        _sample(num_frames=100, fps=30, license="All Rights Reserved"),
    ]
    pipeline = FilterPipeline([
        MinDurationFilter(min_seconds=1.0),
        LicenseFilter(),
    ])
    kept = [pipeline(s) for s in samples]
    assert kept[0] is samples[0]
    assert kept[1] is None
    assert kept[2] is None
    summary = pipeline.summary()
    assert summary["min_duration"]["seen"] == 3
    assert summary["min_duration"]["dropped"] == 1
    # License filter only sees samples that passed duration.
    assert summary["license"]["seen"] == 2
    assert summary["license"]["dropped"] == 1
    assert pipeline.total == 3
    assert pipeline.kept == 1


def test_pipeline_apply_returns_only_kept() -> None:
    samples = [
        _sample(num_frames=100, license="CC-BY"),
        _sample(num_frames=2, license="CC-BY"),
    ]
    pipeline = FilterPipeline([MinDurationFilter(min_seconds=1.0)])
    out = list(pipeline.apply(samples))
    assert len(out) == 1
