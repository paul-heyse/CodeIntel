"""Tests for schema generation, validation, and table creation helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TypedDict

import pyarrow as pa
import pytest

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.schema.json_schema import (
    build_validator,
    export_json_schema_for_contract,
    json_schema_from_typeddict,
    validate_row_with_schema,
)
from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

if typing.TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


class SampleRow(TypedDict):
    """Minimal row model for schema generation tests."""

    name: str
    count: int
    flag: bool


def test_json_schema_from_typeddict_round_trip() -> None:
    """
    Generated schema should validate a conforming row.

    Raises
    ------
    AssertionError
        When schema shape differs from expectations.
    """
    schema = json_schema_from_typeddict(SampleRow)
    if schema["type"] != "object":
        message = "Expected object schema"
        raise AssertionError(message)
    required_raw = typing.cast("list[object]", schema.get("required", []))
    required = {str(key) for key in required_raw}
    if required != {"name", "count", "flag"}:
        message = "Required keys mismatch"
        raise AssertionError(message)
    validate_row_with_schema({"name": "x", "count": 1, "flag": True}, schema)


def test_build_validator_accepts_mapping() -> None:
    """Validator factory should accept mapping schema and expose schema attribute."""
    schema: dict[str, object] = {"type": "object", "properties": {"a": {"type": "string"}}}
    validator = build_validator(schema)
    if getattr(validator, "schema", None) != dict(schema):
        pytest.fail("Validator should retain provided schema mapping")


def test_export_json_schema_prefers_observation() -> None:
    """Observed schemas should win over declared schemas for JSON Schema generation."""
    table_schema = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column(name="id", type="BIGINT", nullable=False),
            Column(name="name", type="VARCHAR", nullable=True),
        ],
    )
    observed_schema = pa.schema([("id", pa.int64())])
    observation = SchemaObservationRecord(
        table_key=table_schema.table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(observed_schema),
    )

    @dataclass(frozen=True)
    class _StubObservationProvider:
        table_key: str
        observation: SchemaObservationRecord

        def load_latest_schema_observation(
            self,
            *,
            table_key: str,
        ) -> SchemaObservationRecord | None:
            if table_key == self.table_key:
                return self.observation
            return None

    contract = DatasetContract(
        table_key=table_schema.table_key,
        name="example",
        schema=table_schema,
        json_schema_id="example",
    )
    schema = export_json_schema_for_contract(
        contract,
        schema_id="urn:test:schema",
        title="Example export",
        observation_provider=_StubObservationProvider(
            table_key=table_schema.table_key,
            observation=observation,
        ),
    )
    if schema is None:
        pytest.fail("Expected JSON Schema from observed schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        pytest.fail("Expected properties mapping in JSON Schema")
    if set(properties) != {"id"}:
        pytest.fail(f"Expected observed-only columns, got: {sorted(properties)}")


def test_validate_row_with_schema_passes_valid_data() -> None:
    """Row validation should pass for conforming data."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    validate_row_with_schema({"a": "ok"}, schema)


def test_apply_all_schemas_creates_function_validation(schema_gateway: StorageGateway) -> None:
    """Schema application should create analytics.function_validation."""
    con = schema_gateway.con
    rows = con.execute("PRAGMA table_info(analytics.function_validation)").fetchall()
    if not rows:
        pytest.fail("analytics.function_validation should exist after apply_all_schemas")
