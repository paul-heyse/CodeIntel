"""Tests for tabular step helpers."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.transforms import tabular_steps


def test_sort_columns_uses_polars_selectors() -> None:
    """Ensure column ordering produces stable output."""
    pl = pytest.importorskip("polars")
    try:
        from polars.testing import assert_frame_equal
    except ImportError:
        assert_frame_equal = pl.testing.assert_frame_equal
    frame = pl.DataFrame({"b": [1, 2], "a": [3, 4]}).lazy()
    sorted_frame = tabular_steps.sort_columns(frame, ["a", "b"])
    result = sorted_frame.collect()
    expected = pl.DataFrame({"a": [3, 4], "b": [1, 2]})
    assert_frame_equal(result, expected)
