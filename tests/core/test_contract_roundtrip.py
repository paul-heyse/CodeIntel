"""Contract schema round-trip tests for JSON/struct columns."""

from __future__ import annotations

from collections.abc import Mapping
import json

import duckdb
import pyarrow as pa
import pytest

from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.hashing.fingerprint import stable_hash
from codeintel.core.iceberg.schema import (
    iceberg_field_ids_for_table_schema,
    table_schema_to_iceberg_schema,
)
from codeintel.core.schemas.contracts import (
    ArrowSchemaMetadata,
    arrow_contract_for_table_schema,
    decode_schema_ipc,
    encode_schema_ipc,
)
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
    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
    aligned_reader = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )
    return pa.Table.from_batches(list(aligned_reader), schema=aligned_reader.schema)


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
        duckdb_reader = relation.fetch_arrow_reader()
        return pa.Table.from_batches(list(duckdb_reader), schema=duckdb_reader.schema)
    finally:
        con.close()


def _decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        try:
            decoded[key_str] = json.loads(raw_str)
        except json.JSONDecodeError:
            decoded[key_str] = raw_str
    return decoded


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


def test_arrow_ipc_roundtrip_preserves_iceberg_metadata() -> None:
    """Arrow IPC round-trip preserves Iceberg field and schema metadata."""
    table_schema = TableSchema(
        schema="analytics",
        name="iceberg_roundtrip",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR", nullable=False),
            Column("payload", "JSON", nullable=True),
        ],
        primary_key=("id",),
    )
    iceberg_bundle = table_schema_to_iceberg_schema(table_schema)
    field_ids = iceberg_field_ids_for_table_schema(table_schema)
    name_mapping_payload = iceberg_bundle.name_mapping.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    name_mapping_digest = stable_hash(name_mapping_payload)
    contract_schema = arrow_contract_for_table_schema(
        table_schema=table_schema,
        metadata=ArrowSchemaMetadata(
            iceberg_schema_id=iceberg_bundle.schema.schema_id,
            iceberg_name_mapping_digest=name_mapping_digest,
            iceberg_field_ids=field_ids,
        ),
    )
    roundtrip = decode_schema_ipc(encode_schema_ipc(contract_schema))
    schema_metadata = _decode_metadata(roundtrip.metadata)
    assert schema_metadata.get("codeintel.iceberg_schema_id") == iceberg_bundle.schema.schema_id
    assert schema_metadata.get("codeintel.iceberg_name_mapping_digest") == name_mapping_digest
    for field in roundtrip:
        field_metadata = _decode_metadata(field.metadata)
        assert field_metadata.get("codeintel.iceberg_field_id") == field_ids[field.name]
