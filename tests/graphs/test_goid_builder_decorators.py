"""Ensure GOID spans include decorator lines."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from codeintel.config import ConfigBuilder, GoidBuilderStepConfig
from codeintel.graphs import goid_builder


def test_goid_start_line_includes_decorator_span() -> None:
    """GOID start_line should widen to the earliest decorator line."""
    row = pd.Series(
        {
            "path": "m.py",
            "node_type": "FunctionDef",
            "name": "foo",
            "qualname": "m.foo",
            "lineno": 10,
            "end_lineno": 12,
            "decorator_start_line": 5,
            "decorator_end_line": 6,
            "col_offset": 0,
            "end_col_offset": 0,
            "parent_qualname": None,
            "decorators": [],
            "docstring": None,
            "hash": "h1",
        }
    )
    cfg = GoidBuilderStepConfig(repo="r", commit="c", language="python")
    now = datetime.now(UTC)

    build_entries = goid_builder._build_goid_entries  # noqa: SLF001
    goid_row, crosswalk_row = build_entries(row, cfg, now, {"m.py": "m"})

    expected_start_line = 5
    if goid_row["start_line"] != expected_start_line:
        message = f"start_line {goid_row['start_line']} != {expected_start_line}"
        pytest.fail(message)
    if crosswalk_row["start_line"] != expected_start_line:
        message = f"crosswalk start_line {crosswalk_row['start_line']} != {expected_start_line}"
        pytest.fail(message)
