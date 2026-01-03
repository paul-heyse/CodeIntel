"""Tests for schema constraint helpers."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.core.columnar.schema_metadata import encode_metadata
from codeintel.core.validation.schema_constraints import schema_metadata_errors


def test_schema_metadata_errors_accepts_valid_metadata() -> None:
    """Valid CodeIntel schema metadata should produce no errors."""
    metadata = encode_metadata(
        {
            "codeintel.table_key": "analytics.demo",
            "codeintel.schema_hash": "hash",
            "codeintel.schema_digest": "digest",
            "codeintel.primary_key": ["id"],
            "codeintel.schema_contract_version": "v1",
            "codeintel.extras_policy": "retain",
            "codeintel.extras_column": "_ci_extras",
            "codeintel.extras_schema": {"extra": "JSON"},
            "codeintel.description": "Demo schema",
            "codeintel.provenance": {"source": "tests"},
        }
    )
    schema = pa.schema([pa.field("id", pa.int64())], metadata=metadata)

    errors = schema_metadata_errors(schema)
    if errors:
        pytest.fail(f"Expected no metadata errors, got {errors}")


def test_schema_metadata_errors_flags_invalid_values() -> None:
    """Invalid metadata types and unknown keys should be reported."""
    metadata = encode_metadata(
        {
            "codeintel.table_key": 123,
            "codeintel.primary_key": "id",
            "codeintel.extras_policy": "unsupported",
            "codeintel.extras_schema": ["bad"],
            "codeintel.unknown_key": "value",
        }
    )
    schema = pa.schema([pa.field("id", pa.int64())], metadata=metadata)
    errors = schema_metadata_errors(schema)

    if not any("Unknown schema metadata key" in error for error in errors):
        pytest.fail("Expected unknown key error")
    if not any("codeintel.table_key" in error for error in errors):
        pytest.fail("Expected table_key type error")
    if not any("codeintel.primary_key" in error for error in errors):
        pytest.fail("Expected primary_key type error")
    if not any("codeintel.extras_policy" in error for error in errors):
        pytest.fail("Expected extras_policy error")
    if not any("codeintel.extras_schema" in error for error in errors):
        pytest.fail("Expected extras_schema error")
