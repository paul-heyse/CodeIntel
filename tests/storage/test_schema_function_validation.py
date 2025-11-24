"""Ensure schema application creates the function_validation table."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import open_memory_gateway


def test_apply_all_schemas_creates_function_validation() -> None:
    """Schema application should create analytics.function_validation."""
    con = open_memory_gateway(apply_schema=True).con
    rows = con.execute("PRAGMA table_info(analytics.function_validation)").fetchall()
    if not rows:
        pytest.fail("analytics.function_validation should exist after apply_all_schemas")
