"""Ensure schema application creates the function_validation table."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.storage.schemas import apply_all_schemas


def test_apply_all_schemas_creates_function_validation() -> None:
    """Schema application should create analytics.function_validation."""
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)
    rows = con.execute("PRAGMA table_info(analytics.function_validation)").fetchall()
    if not rows:
        pytest.fail("analytics.function_validation should exist after apply_all_schemas")
