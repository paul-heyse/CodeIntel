"""Tests for schema-derived row serialization."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import configure_schema_service, get_schema_provider
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.runtime.runtime_bundle import RuntimeBundle


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def test_row_serializer_matches_schema_order() -> None:
    """Row serializers should follow TableSchema column ordering."""
    table_key = "core.modules"
    schema = get_schema_provider().require_table_schema(table_key)
    column_names = schema.column_names()
    row = {name: f"value_{idx}" for idx, name in enumerate(column_names)}

    serializer = row_serializer_for_table_key(table_key)
    result = serializer(row)
    expected = tuple(row[name] for name in column_names)

    if result != expected:
        pytest.fail("Row serializer order should match TableSchema column order.")
