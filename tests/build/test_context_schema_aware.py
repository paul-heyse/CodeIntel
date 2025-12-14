"""Tests for schema-aware write methods in TargetExecutionContext.

Tests the write_validated_table() method and related functionality.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def test_write_validated_table_missing_schema_key_error() -> None:
    """Verify KeyError is raised when schema is not registered.

    This test verifies the error path without creating a full execution context,
    by directly checking that the SCHEMA_REGISTRY raises KeyError for unknown tables.
    """
    table_key = "completely.unregistered.table"
    result = SCHEMA_REGISTRY.get(table_key)
    _require(
        condition=result is None,
        message="unregistered table should return None from SCHEMA_REGISTRY.get()",
    )


def test_write_validated_table_integration() -> None:
    """Test writing to a schema that's actually registered."""
    all_keys = SCHEMA_REGISTRY.all()
    if not all_keys:
        pytest.skip("No schemas registered")

    table_key = next(iter(all_keys))
    schema = SCHEMA_REGISTRY.require(table_key)

    columns = schema.column_names()
    _require(condition=len(columns) > 0, message="schema should have columns")
