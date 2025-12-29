"""Tests for Arrow/Polars schema conversions."""

from __future__ import annotations

import json

import polars as pl
import pyarrow as pa
import pytest

from codeintel.core.schemas.contracts import (
    ARROW_SCHEMA_CONTRACT_VERSION,
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    arrow_contract_for_table_schema,
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
)
from codeintel.core.schemas.primitives import Column, TableSchema


def _encode_metadata(metadata: dict[str, object]) -> dict[bytes, bytes]:
    encoded: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            raw = value
        else:
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        encoded[key.encode("utf-8")] = raw.encode("utf-8")
    return encoded


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, object]:
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


def test_arrow_schema_maps_types_and_nullability() -> None:
    """Verify Arrow types + nullability map to TableSchema."""
    fields = [
        pa.field("flag", pa.bool_(), nullable=False),
        pa.field("int32_col", pa.int32(), nullable=False),
        pa.field("int64_col", pa.int64(), nullable=True),
        pa.field("uint64_col", pa.uint64(), nullable=True),
        pa.field("float_col", pa.float64(), nullable=True),
        pa.field("decimal_col", pa.decimal128(10, 2), nullable=True),
        pa.field("decimal38_col", pa.decimal128(38, 0), nullable=False),
        pa.field("timestamp_col", pa.timestamp("us"), nullable=True),
        pa.field("timestamptz_col", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("string_col", pa.string(), nullable=True),
        pa.field("string_view_col", pa.string_view(), nullable=True),
        pa.field("binary_col", pa.binary(), nullable=True),
        pa.field("struct_col", pa.struct([("a", pa.int32())]), nullable=True),
    ]
    schema = pa.schema(fields)
    table_schema = table_schema_from_arrow_schema(
        arrow_schema=schema,
        table_key="analytics.demo",
    )

    expected = [
        ("flag", "BOOLEAN", False),
        ("int32_col", "INTEGER", False),
        ("int64_col", "BIGINT", True),
        ("uint64_col", "DECIMAL(38,0)", True),
        ("float_col", "DOUBLE", True),
        ("decimal_col", "DECIMAL", True),
        ("decimal38_col", "DECIMAL(38,0)", False),
        ("timestamp_col", "TIMESTAMP", True),
        ("timestamptz_col", "TIMESTAMPTZ", False),
        ("string_col", "VARCHAR", True),
        ("string_view_col", "VARCHAR", True),
        ("binary_col", "VARCHAR", True),
        ("struct_col", "JSON", True),
    ]
    actual = [(col.name, col.type, col.nullable) for col in table_schema.columns]
    if actual != expected:
        pytest.fail(f"Column mapping mismatch: {actual} != {expected}")


def test_arrow_schema_respects_metadata_overrides() -> None:
    """Verify schema/field metadata overrides are respected."""
    fields = [
        pa.field(
            "id",
            pa.int32(),
            nullable=False,
            metadata=_encode_metadata(
                {
                    "codeintel.column_type": "BIGINT",
                    "codeintel.description": "Identifier",
                }
            ),
        ),
        pa.field("source_id", pa.string(), nullable=True),
    ]
    schema_metadata = _encode_metadata(
        {
            "codeintel.table_key": "analytics.demo",
            "codeintel.description": "Demo table",
            "codeintel.primary_key": ["id", "source_id"],
        }
    )
    schema = pa.schema(fields, metadata=schema_metadata)
    table_schema = table_schema_from_arrow_schema(arrow_schema=schema)

    if table_schema.table_key != "analytics.demo":
        pytest.fail(f"Unexpected table_key: {table_schema.table_key}")
    if table_schema.description != "Demo table":
        pytest.fail("Expected table description from schema metadata")
    if table_schema.primary_key != ("id", "source_id"):
        pytest.fail("Expected primary_key from schema metadata")

    id_column = table_schema.columns[0]
    if id_column.type != "BIGINT":
        pytest.fail("Expected column_type override from field metadata")
    if id_column.description != "Identifier":
        pytest.fail("Expected column description from field metadata")


def test_arrow_schema_uses_key_role_fallback() -> None:
    """Verify key-role metadata drives primary key fallback."""
    fields = [
        pa.field(
            "id",
            pa.int64(),
            nullable=False,
            metadata=_encode_metadata({"codeintel.key_role": "primary_key"}),
        ),
        pa.field(
            "name",
            pa.string(),
            nullable=True,
            metadata=_encode_metadata({"codeintel.key_role": "primary_key"}),
        ),
    ]
    schema = pa.schema(fields, metadata=_encode_metadata({"codeintel.table_key": "core.tags"}))
    table_schema = table_schema_from_arrow_schema(arrow_schema=schema)

    if table_schema.primary_key != ("id", "name"):
        pytest.fail(f"Expected key_role-derived primary_key, got {table_schema.primary_key}")


def test_polars_dataframe_schema_conversion() -> None:
    """Verify Polars DataFrame schemas convert to TableSchema."""
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["a", "b"],
            "active": [True, False],
        }
    )
    table_schema = table_schema_from_polars_dataframe(
        frame=frame,
        table_key="analytics.polars_demo",
    )
    expected = [("id", "BIGINT"), ("name", "VARCHAR"), ("active", "BOOLEAN")]
    actual = [(col.name, col.type) for col in table_schema.columns]
    if actual != expected:
        pytest.fail(f"Unexpected Polars schema mapping: {actual}")


def test_arrow_contract_roundtrip_preserves_table_schema() -> None:
    """Arrow contracts should round-trip back into the original TableSchema."""
    table_schema = TableSchema(
        schema="analytics",
        name="arrow_contract_roundtrip",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR", nullable=True),
            Column("score", "DOUBLE", nullable=True),
        ],
        primary_key=("id",),
        description="Arrow contract roundtrip test",
    )

    contract_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    roundtrip = table_schema_from_arrow_schema(arrow_schema=contract_schema)
    if roundtrip.to_json_obj() != table_schema.to_json_obj():
        pytest.fail("Arrow contract roundtrip did not preserve TableSchema")

    metadata = _decode_metadata(contract_schema.metadata)
    if metadata.get("codeintel.schema_contract_version") != ARROW_SCHEMA_CONTRACT_VERSION:
        pytest.fail("Arrow contract schema_contract_version metadata mismatch")
    if metadata.get("codeintel.extras_policy") != DEFAULT_EXTRAS_POLICY:
        pytest.fail("Arrow contract extras_policy metadata mismatch")
    if metadata.get("codeintel.extras_column") != DEFAULT_EXTRAS_COLUMN:
        pytest.fail("Arrow contract extras_column metadata mismatch")
    if metadata.get("codeintel.table_key") != table_schema.table_key:
        pytest.fail("Arrow contract table_key metadata mismatch")
