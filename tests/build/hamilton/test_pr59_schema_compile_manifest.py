"""PR-59: Schema manifest compilation and determinism."""

from __future__ import annotations

import json

import pytest

from codeintel.build.schemas.compile import compile_schema_manifest_for_table_keys
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider


def test_compile_manifest_orders_tables_by_table_key() -> None:
    """Manifest compilation should produce deterministic table ordering."""
    schema_a = TableSchema(schema="analytics", name="a", columns=[Column("repo", "VARCHAR")])
    schema_b = TableSchema(schema="analytics", name="b", columns=[Column("repo", "VARCHAR")])
    provider = MappingSchemaProvider({"analytics.b": schema_b, "analytics.a": schema_a})

    manifest = compile_schema_manifest_for_table_keys(
        ["analytics.b", "analytics.a"],
        provider=provider,
    )

    actual = [t.table_key for t in manifest.tables]
    expected = ["analytics.a", "analytics.b"]
    if actual != expected:
        pytest.fail(f"Unexpected table ordering: {actual} != {expected}")


def test_manifest_json_roundtrip_is_valid_json() -> None:
    """Manifest JSON object should be JSON-serializable."""
    schema = TableSchema(schema="core", name="example", columns=[Column("id", "BIGINT")])
    provider = MappingSchemaProvider({"core.example": schema})
    manifest = compile_schema_manifest_for_table_keys(["core.example"], provider=provider)

    payload = json.dumps(manifest.to_json_obj(), sort_keys=True)
    reloaded = json.loads(payload)
    if not isinstance(reloaded, dict):
        pytest.fail("Expected manifest JSON to decode into an object")
    if reloaded.get("version") != manifest.version:
        pytest.fail("Manifest version did not round-trip through JSON")
    tables = reloaded.get("tables")
    if not isinstance(tables, list) or len(tables) != 1:
        pytest.fail("Expected manifest JSON to include exactly one table entry")
