"""Tests for dataset column serialization helpers."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import configure_schema_service
from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers.assertions.expectation_assertions import expect_equal


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def test_serialize_row_uses_schema_when_table_key_provided() -> None:
    """serialize_row should delegate to schema-backed ordering with a table key."""
    table_key = "core.modules"
    columns = load_columns_by_table()[table_key]
    row = {col: f"value_{idx}" for idx, col in enumerate(columns)}

    expected = row_to_tuple(table_key, row)
    result = serialize_row(row, columns, table_key=table_key)

    expect_equal(result, expected)


def test_serialize_row_uses_columns_without_table_key() -> None:
    """serialize_row should honor provided columns when table_key is missing."""
    row = {"a": 1, "b": 2}
    result = serialize_row(row, ["b", "a"], table_key=None)
    expect_equal(result, (2, 1))


def test_serialize_row_requires_columns_without_table_key() -> None:
    """serialize_row should error when columns are omitted without a table key."""
    with pytest.raises(ValueError, match="columns must be provided"):
        serialize_row({"a": 1}, None)
