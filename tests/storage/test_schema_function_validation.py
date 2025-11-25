"""Ensure schema application creates the function_validation table."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import StorageGateway


def test_apply_all_schemas_creates_function_validation(fresh_gateway: StorageGateway) -> None:
    """Schema application should create analytics.function_validation."""
    con = fresh_gateway.con
    rows = con.execute("PRAGMA table_info(analytics.function_validation)").fetchall()
    if not rows:
        pytest.fail("analytics.function_validation should exist after apply_all_schemas")
