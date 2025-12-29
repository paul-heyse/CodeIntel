"""Tests for schema catalog persistence and cache prefill."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pytest

from codeintel.build.schemas.inference_service import SchemaInferenceService
from codeintel.build.schemas.manifest import SchemaManifest, TableProvenance
from codeintel.build.schemas.schema_index import SchemaDerivation, SchemaIndex
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.tracking.schema_catalog import SchemaCatalogRequest
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


def _provenance_for_schema(table_schema: TableSchema) -> TableProvenance:
    return TableProvenance(
        schema_hash=schema_hash(table_schema),
        derivation_kind="inferred_relation",
        derivation_source="test",
        inference_status="inferred",
    )


def _decode_arrow_schema(payload: str) -> pa.Schema:
    raw = base64.b64decode(payload)
    buffer = pa.py_buffer(raw)
    return pa.ipc.read_schema(pa.BufferReader(buffer))


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
        request=SchemaCatalogRequest(
            run_id="run-1",
            repo="org/repo",
            commit="deadbeef",
        ),
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
                derivation_kind="inferred_relation",
                derivation_source="test",
                inference_status="inferred",
            )
        },
    )

    fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        request=SchemaCatalogRequest(
            run_id="run-2",
            repo="org/repo",
            commit="deadbeef",
        ),
    )

    schema_index = SchemaIndex(
        derivations={
            table_key: SchemaDerivation(
                table_key=table_key,
                kind="inferred_relation",
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


def test_override_registry_refresh_populates_registry(
    fresh_gateway: StorageGateway,
) -> None:
    """Override refresh should populate registry for inferred tables."""
    table_key = "analytics.override_table"
    table_schema = _schema_for_key(table_key)
    manifest = SchemaManifest(
        version="v2",
        tables=(table_schema,),
        table_provenance={table_key: _provenance_for_schema(table_schema)},
    )

    request = SchemaCatalogRequest(
        run_id="run-override-1",
        repo="org/repo",
        commit="deadbeef",
    )
    persist_result = fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        request=request,
    )
    override_result = fresh_gateway.schemas.refresh_override_registry_from_manifest(
        manifest,
        request=request,
        catalog_hash=persist_result.catalog_hash,
    )

    expect_equal(override_result.status, "updated", label="override_status")
    expect_equal(override_result.tables, 1, label="override_tables")
    expect_is_not_none(override_result.version_id, label="override_version_id")

    overrides = fresh_gateway.schemas.load_override_registry()
    override_schema = expect_is_not_none(
        overrides.get(table_key),
        label="override_schema",
    )
    expect_equal(
        override_schema.to_json_obj(),
        table_schema.to_json_obj(),
        label="override_schema_roundtrip",
    )


def test_override_registry_pin_restores_prior_version(
    fresh_gateway: StorageGateway,
) -> None:
    """Override pin should restore a previous override version."""
    table_key = "analytics.override_pin"
    schema_v1 = _schema_for_key(table_key)
    schema_v2 = TableSchema(
        schema=schema_v1.schema,
        name=schema_v1.name,
        columns=[
            *schema_v1.columns,
            Column("name", "VARCHAR"),
        ],
    )

    request = SchemaCatalogRequest(
        run_id="run-override-2",
        repo="org/repo",
        commit="deadbeef",
    )

    manifest_v1 = SchemaManifest(
        version="v2",
        tables=(schema_v1,),
        table_provenance={table_key: _provenance_for_schema(schema_v1)},
    )
    persist_v1 = fresh_gateway.schemas.persist_schema_manifest(
        manifest_v1,
        request=request,
    )
    override_v1 = fresh_gateway.schemas.refresh_override_registry_from_manifest(
        manifest_v1,
        request=request,
        catalog_hash=persist_v1.catalog_hash,
    )
    version_id_v1 = expect_is_not_none(override_v1.version_id, label="version_id_v1")

    manifest_v2 = SchemaManifest(
        version="v2",
        tables=(schema_v2,),
        table_provenance={table_key: _provenance_for_schema(schema_v2)},
    )
    persist_v2 = fresh_gateway.schemas.persist_schema_manifest(
        manifest_v2,
        request=request,
    )
    fresh_gateway.schemas.refresh_override_registry_from_manifest(
        manifest_v2,
        request=request,
        catalog_hash=persist_v2.catalog_hash,
    )

    overrides_latest = fresh_gateway.schemas.load_override_registry()
    latest_schema = expect_is_not_none(
        overrides_latest.get(table_key),
        label="latest_override_schema",
    )
    expect_equal(
        latest_schema.to_json_obj(),
        schema_v2.to_json_obj(),
        label="override_schema_latest",
    )

    pinned_record = fresh_gateway.schemas.set_override_registry_version(
        table_key=table_key,
        version_id=version_id_v1,
    )
    expect_equal(pinned_record.version_id, version_id_v1, label="pinned_version_id")

    overrides_pinned = fresh_gateway.schemas.load_override_registry()
    pinned_schema = expect_is_not_none(
        overrides_pinned.get(table_key),
        label="pinned_override_schema",
    )
    expect_equal(
        pinned_schema.to_json_obj(),
        schema_v1.to_json_obj(),
        label="override_schema_pinned",
    )


def test_registry_health_snapshot_reflects_latest_manifest(
    fresh_gateway: StorageGateway,
) -> None:
    """Registry health should reflect the latest manifest and overrides."""
    table_keys = {
        "inferable": "analytics.health_inferable",
        "explicit": "analytics.health_explicit",
        "view": "docs.v_health_view",
    }
    schemas = {name: _schema_for_key(key) for name, key in table_keys.items()}

    manifest = SchemaManifest(
        version="v2",
        tables=(schemas["explicit"], schemas["inferable"]),
        views=(schemas["view"],),
        table_provenance={
            table_keys["explicit"]: TableProvenance(
                schema_hash=schema_hash(schemas["explicit"]),
                derivation_kind="explicit_override",
                derivation_source="test",
            ),
            table_keys["inferable"]: _provenance_for_schema(schemas["inferable"]),
        },
        view_provenance={
            table_keys["view"]: TableProvenance(
                schema_hash=schema_hash(schemas["view"]),
                derivation_kind="view_inferred",
                derivation_source="duckdb",
            ),
        },
    )

    request = SchemaCatalogRequest(
        run_id="run-health-1",
        repo="org/repo",
        commit="deadbeef",
    )
    persist_result = fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        request=request,
    )
    refresh_result = fresh_gateway.schemas.refresh_override_registry_from_manifest(
        manifest,
        request=request,
        catalog_hash=persist_result.catalog_hash,
    )

    health = fresh_gateway.schemas.registry_health_snapshot()
    expect_equal(cast("str", health.get("status")), "ok", label="health_status")

    latest_manifest = expect_is_not_none(
        cast("dict[str, object] | None", health.get("latest_manifest")),
        label="latest_manifest",
    )
    expect_equal(
        latest_manifest.get("catalog_hash"),
        persist_result.catalog_hash,
        label="latest_manifest_hash",
    )
    expect_equal(
        latest_manifest.get("repo"),
        request.repo,
        label="latest_manifest_repo",
    )
    expect_equal(
        latest_manifest.get("commit"),
        request.commit,
        label="latest_manifest_commit",
    )

    expect_equal(cast("int", health.get("registry_rows")), 3, label="registry_rows")
    expect_equal(
        cast("int", health.get("override_registry_rows")),
        refresh_result.tables,
        label="override_registry_rows",
    )
    expect_equal(cast("int", health.get("inferable_total")), 1, label="inferable_total")
    expect_equal(cast("int", health.get("inferred_count")), 1, label="inferred_count")
    expect_equal(
        cast("int", health.get("inference_error_count")),
        0,
        label="inference_error_count",
    )
    expect_equal(
        cast("float | None", health.get("inference_success_rate")),
        1.0,
        label="inference_success_rate",
    )
    expect_equal(
        cast("bool", health.get("registry_stale")),
        expected=False,
        label="registry_stale",
    )

    for table_key in table_keys.values():
        loaded_schema = fresh_gateway.schemas.load_table_schema(table_key)
        expect_is_not_none(loaded_schema, label=f"schema_digest_{table_key}")


def test_schema_versions_persist_arrow_contract_payload(
    fresh_gateway: StorageGateway,
) -> None:
    """Schema versions should store Arrow contract payloads in renderer_cache."""
    table_key = "analytics.arrow_contract_payload"
    table_schema = _schema_for_key(table_key)
    manifest = SchemaManifest(
        version="v2",
        tables=(table_schema,),
        table_provenance={table_key: _provenance_for_schema(table_schema)},
    )

    fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        request=SchemaCatalogRequest(
            run_id="run-arrow-contract",
            repo="org/repo",
            commit="deadbeef",
        ),
    )

    schema_digest = fingerprint(table_schema.to_json_obj())
    versions_ref = meta_table_ref("metadata.schema_versions")
    row = expect_is_not_none(
        fresh_gateway.con.execute(
            f"SELECT renderer_cache FROM {versions_ref} WHERE schema_digest = ?",
            [schema_digest],
        ).fetchone(),
        label="renderer_cache_row",
    )
    renderer_cache = decode_json_dict(row[0])
    payload = renderer_cache.get("arrow_schema_ipc_b64")
    if not isinstance(payload, str):
        pytest.fail("Expected arrow_schema_ipc_b64 in renderer_cache")

    schema = _decode_arrow_schema(payload)
    metadata = schema.metadata or {}
    schema_hash_value = metadata.get(b"codeintel.schema_hash")
    if schema_hash_value is None or schema_hash_value.decode("utf-8") != schema_hash(table_schema):
        pytest.fail("Arrow contract schema_hash metadata mismatch")
    schema_digest_value = metadata.get(b"codeintel.schema_digest")
    if schema_digest_value is None or schema_digest_value.decode("utf-8") != schema_digest:
        pytest.fail("Arrow contract schema_digest metadata mismatch")


def test_schema_versions_backfill_renderer_cache(
    fresh_gateway: StorageGateway,
) -> None:
    """Backfill should populate missing renderer_cache entries."""
    table_key = "analytics.arrow_contract_backfill"
    table_schema = _schema_for_key(table_key)
    manifest = SchemaManifest(
        version="v2",
        tables=(table_schema,),
        table_provenance={table_key: _provenance_for_schema(table_schema)},
    )

    fresh_gateway.schemas.persist_schema_manifest(
        manifest,
        request=SchemaCatalogRequest(
            run_id="run-arrow-backfill",
            repo="org/repo",
            commit="deadbeef",
        ),
    )

    schema_digest = fingerprint(table_schema.to_json_obj())
    versions_ref = meta_table_ref("metadata.schema_versions")
    fresh_gateway.con.execute(
        f"UPDATE {versions_ref} SET renderer_cache = NULL WHERE schema_digest = ?",
        [schema_digest],
    )

    updated = fresh_gateway.schemas.backfill_renderer_cache(manifest)
    expect_equal(updated, 1, label="backfill_rows")

    row = expect_is_not_none(
        fresh_gateway.con.execute(
            f"SELECT renderer_cache FROM {versions_ref} WHERE schema_digest = ?",
            [schema_digest],
        ).fetchone(),
        label="renderer_cache_row",
    )
    renderer_cache = decode_json_dict(row[0])
    payload = renderer_cache.get("arrow_schema_ipc_b64")
    if not isinstance(payload, str) or not payload:
        pytest.fail("Expected arrow_schema_ipc_b64 after backfill")
