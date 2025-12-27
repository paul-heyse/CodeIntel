"""Tests for schema catalog persistence and cache prefill."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.schemas.inference_service import SchemaInferenceService
from codeintel.build.schemas.manifest import SchemaManifest, TableProvenance
from codeintel.build.schemas.schema_index import SchemaDerivation, SchemaIndex
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.helpers.table_key import split_table_key
from tests._helpers.assertions import expect_equal, expect_is_none, expect_is_not_none

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def _schema_for_key(table_key: str) -> TableSchema:
    schema_name, table_name = split_table_key(table_key)
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=[Column("id", "BIGINT", nullable=False)],
    )


def test_schema_manifest_roundtrip_from_metadata(fresh_gateway: StorageGateway) -> None:
    """Persisted schema manifests should round-trip through metadata tables."""
    table_key = "analytics.roundtrip_table"
    view_key = "docs.v_roundtrip_view"
    table_schema = _schema_for_key(table_key)
    view_schema = _schema_for_key(view_key)

    manifest = SchemaManifest(
        version="v2",
        tables=(table_schema,),
        views=(view_schema,),
        table_provenance={
            table_key: TableProvenance(
                schema_hash=schema_hash(table_schema),
                derivation_kind="explicit_override",
                derivation_source="test",
            )
        },
        view_provenance={
            view_key: TableProvenance(
                schema_hash=schema_hash(view_schema),
                derivation_kind="view_inferred",
                derivation_source="duckdb",
            )
        },
    )

    result = fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        run_id="run-1",
        repo="org/repo",
        commit="deadbeef",
    )

    expect_equal(result.tables, 1, label="tables")
    expect_equal(result.views, 1, label="views")
    expect_equal(result.schema_manifest_runs_rows, 1, label="manifest_runs")
    expect_equal(result.schema_versions_rows, 2, label="schema_versions")
    expect_equal(result.table_schema_registry_rows, 2, label="table_schema_registry")

    loaded_table = expect_is_not_none(
        fresh_gateway.schemas.load_table_schema(table_key),
        label="table_schema",
    )
    loaded_view = expect_is_not_none(
        fresh_gateway.schemas.load_table_schema(view_key),
        label="view_schema",
    )

    expect_equal(loaded_table.to_json_obj(), table_schema.to_json_obj(), label="table_roundtrip")
    expect_equal(loaded_view.to_json_obj(), view_schema.to_json_obj(), label="view_roundtrip")


def test_schema_index_prefill_avoids_inference(fresh_gateway: StorageGateway) -> None:
    """Prefill should seed SchemaIndex cache for inferred tables."""

    class _FailingInferenceService:
        @staticmethod
        def infer_table_schema(table_key: str, *, declared_provider: object) -> TableSchema:
            _ = declared_provider
            msg = f"Unexpected inference for {table_key}"
            raise AssertionError(msg)

    table_key = "analytics.prefill_cache"
    table_schema = _schema_for_key(table_key)
    manifest = SchemaManifest(
        version="v2",
        tables=(table_schema,),
        table_provenance={
            table_key: TableProvenance(
                schema_hash=schema_hash(table_schema),
                derivation_kind="inferred_ibis",
                derivation_source="test",
                inference_status="inferred",
            )
        },
    )

    fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        run_id="run-2",
        repo="org/repo",
        commit="deadbeef",
    )

    schema_index = SchemaIndex(
        derivations={
            table_key: SchemaDerivation(
                table_key=table_key,
                kind="inferred_ibis",
                source="test",
                override_schema=None,
            )
        },
        inferable_table_keys=frozenset({table_key}),
        declared_provider=MappingSchemaProvider({}),
        inference_service=cast("SchemaInferenceService", _FailingInferenceService()),
    )

    prefilled = fresh_gateway.schemas.prefill_schema_index(schema_index)
    expect_equal(prefilled, 1, label="prefilled_count")

    resolved = schema_index.get_table_schema(table_key)
    expect_is_none(schema_index.get_inference_error(table_key), label="inference_error")
    resolved_schema = expect_is_not_none(resolved, label="resolved_schema")
    expect_equal(
        resolved_schema.to_json_obj(),
        table_schema.to_json_obj(),
        label="prefill_schema",
    )
