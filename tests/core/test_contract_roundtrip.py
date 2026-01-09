"""Contract schema round-trip tests for JSON/struct columns."""

from __future__ import annotations

import json

import duckdb
import pyarrow as pa
import pytest

from codeintel.core.columnar.conversion import (
    reader_from_batches,
    table_from_batches,
    tabular_to_arrow_table,
)
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import Column, TableSchema


def _contract_schema() -> pa.Schema:
    table_schema = TableSchema(
        schema="analytics",
        name="json_roundtrip",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("payload", "JSON", nullable=True),
        ],
        primary_key=("id",),
    )
    return arrow_contract_for_table_schema(table_schema=table_schema)


def _expected_payload() -> list[str | None]:
    return [
        json.dumps({"a": 1}, sort_keys=True, separators=(",", ":")),
        None,
    ]


def _is_string_type(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _aligned_table(contract_schema: pa.Schema) -> pa.Table:
    struct_array = pa.array(
        [{"a": 1}, None],
        type=pa.struct([("a", pa.int64())]),
    )
    batch = pa.record_batch(
        [pa.array([1, 2]), struct_array],
        names=["id", "payload"],
    )
    reader = reader_from_batches(batch.schema, [batch])
    aligned_reader = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )
    return table_from_batches(aligned_reader, schema=aligned_reader.schema)


def _assert_payload_table(table: pa.Table, *, label: str) -> None:
    payload_values = table.column("payload").to_pylist()
    if payload_values != _expected_payload():
        pytest.fail(f"{label} payload mismatch: {payload_values}")
    if not _is_string_type(table.schema.field("payload").type):
        pytest.fail(f"{label} payload column is not a string type")


def _duckdb_table(aligned_table: pa.Table) -> pa.Table:
    con = duckdb.connect()
    try:
        relation = con.from_arrow(aligned_table)
        return tabular_to_arrow_table(relation)
    finally:
        con.close()


def test_json_struct_roundtrip_polars_duckdb() -> None:
    """JSON contracts should round-trip through Polars and DuckDB as strings."""
    pl = pytest.importorskip("polars")
    contract_schema = _contract_schema()

    aligned_table = _aligned_table(contract_schema)
    _assert_payload_table(aligned_table, label="Aligned")
    polars_table = pl.from_arrow(aligned_table).to_arrow()
    _assert_payload_table(polars_table, label="Polars")
    duckdb_table = _duckdb_table(aligned_table)
    _assert_payload_table(duckdb_table, label="DuckDB")
