"""Tests for the package metadata."""

from __future__ import annotations

import another_world


def test_version_is_set() -> None:
    assert isinstance(another_world.__version__, str)
    assert another_world.__version__.count(".") >= 1
