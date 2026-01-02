"""PR-58: Build fingerprinting uses SchemaProvider-driven schema hashing."""

from __future__ import annotations

import pytest

from codeintel.build.assets.fingerprinting import compute_table_schema_hash
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider


def test_compute_table_schema_hash_delegates_to_schema_provider() -> None:
    """compute_table_schema_hash should use the resolved TableSchema from provider."""
    table_schema = TableSchema(
        schema="analytics",
        name="function_types",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("loc", "INTEGER"),
        ],
    )
    provider = MappingSchemaProvider({"analytics.function_types": table_schema})

    actual = compute_table_schema_hash(
        "analytics.function_types",
        schema_provider=provider,
    )
    expected = schema_hash(table_schema)
    if actual != expected:
        pytest.fail(f"Expected schema hash {expected}, got {actual}")


def test_compute_table_schema_hash_returns_none_when_unknown() -> None:
    """compute_table_schema_hash should return None when no schema exists."""
    provider = MappingSchemaProvider({})
    missing = compute_table_schema_hash("analytics.missing_table", schema_provider=provider)
    if missing is not None:
        pytest.fail("Expected None for unknown table schema hash")
