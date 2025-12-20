"""Tests for advanced type support in schema round-trip helpers."""

from __future__ import annotations

from typing import cast

from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.schema_roundtrip import create_table_ast, ibis_schema_from_table_schema
from tests._helpers.assertions.expectation_assertions import expect_true


def test_schema_roundtrip_supports_advanced_types() -> None:
    """Round-trip helpers support nested/advanced DuckDB types."""
    schema = TableSchema(
        schema="core",
        name="advanced_types",
        columns=[
            Column("id", cast("ColumnType", "UUID"), nullable=False),
            Column("tags", cast("ColumnType", "LIST<VARCHAR>")),
            Column("attrs", cast("ColumnType", "MAP<VARCHAR, INTEGER>")),
            Column("payload", cast("ColumnType", "STRUCT<id INTEGER, label VARCHAR>")),
            Column("ts_ns", cast("ColumnType", "TIMESTAMP_NS")),
        ],
    )

    ibis_schema = ibis_schema_from_table_schema(schema)
    expect_true(ibis_schema["id"].is_uuid(), message="uuid type")
    expect_true(ibis_schema["tags"].is_array(), message="list/array type")
    expect_true(ibis_schema["attrs"].is_map(), message="map type")
    expect_true(ibis_schema["payload"].is_struct(), message="struct type")
    expect_true(ibis_schema["ts_ns"].is_timestamp(), message="timestamp type")

    ddl = create_table_ast(schema, if_not_exists=True).sql(dialect=DUCKDB_DIALECT)
    expect_true("UUID" in ddl, message="uuid ddl")
    expect_true("TEXT[]" in ddl, message="array ddl")
    expect_true("MAP(" in ddl, message="map ddl")
    expect_true("STRUCT" in ddl, message="struct ddl")
