"""IPC streaming tests for the semantic query kernel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from codeintel.core.manifests import SchemaManifest
from codeintel.core.schemas import table_schema_from_json_obj
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.gateway.pool import PoolConfig
from codeintel.storage.schema import arrow_schema_for_table_key
from codeintel.storage.schema.duckdb_contracts import ContractSchemaOptions
from codeintel.storage.tracking.schema_catalog import SchemaCatalogRequest
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.gateway import seed_contract_catalog
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.serving_snapshot_factory import ServingSnapshot


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def _decode_metadata_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _schema_manifest_from_path(path: Path) -> SchemaManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "schema_manifest payload must be a JSON object"
        raise TypeError(msg)
    version = str(payload.get("version", "")).strip()
    tables_raw = payload.get("tables", [])
    views_raw = payload.get("views", [])
    if not isinstance(tables_raw, list) or not isinstance(views_raw, list):
        msg = "schema_manifest tables/views must be arrays"
        raise TypeError(msg)
    tables = tuple(table_schema_from_json_obj(item) for item in tables_raw)
    views = tuple(table_schema_from_json_obj(item) for item in views_raw)
    return SchemaManifest(version=version, tables=tables, views=views)


def _seed_contract_catalog(
    snapshot: ServingSnapshot,
    *,
    db_path: Path,
    repo: str,
    commit: str,
) -> None:
    config = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        repo=repo,
        commit=commit,
    )
    gateway = open_gateway(config, seed_contract_catalog=seed_contract_catalog)
    try:
        manifest = _schema_manifest_from_path(snapshot.schema_manifest_path)
        gateway.schemas.persist_schema_manifest(
            manifest,
            request=SchemaCatalogRequest(
                run_id="run-ipc-stream",
                repo=repo,
                commit=commit,
                strict_provenance=False,
            ),
        )
    finally:
        gateway.close()


def _load_contract_schema(
    *,
    snapshot: ServingSnapshot,
    table_key: str,
) -> pa.Schema:
    config = StorageConfig(
        db_path=snapshot.db_path,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    gateway = open_gateway(config)
    try:
        options = ContractSchemaOptions(
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        schema = arrow_schema_for_table_key(
            gateway.con,
            table_key=table_key,
            options=options,
        )
        if schema is None:
            pytest.fail(f"Expected contract schema for {table_key}")
        return schema
    finally:
        gateway.close()


@pytest.mark.anyio
async def test_query_ipc_stream_includes_metadata_and_rows(tmp_path: Path) -> None:
    """Arrow IPC stream includes schema metadata and expected rows."""
    snapshot = ServingSnapshotFactory(tmp_path).demo_snapshot(row_count=3)
    _seed_contract_catalog(
        snapshot,
        db_path=snapshot.db_path,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    contract_schema = _load_contract_schema(snapshot=snapshot, table_key="docs.v_demo")

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="polars",
                schema_enforcement="strict",
            ),
        )

        request = SemanticQueryRequest(
            view_id="demo.view",
            filters=[],
            order_by=["id"],
            limit=2,
            offset=0,
        )
        stream = kernel.query_ipc_stream(request)
        data = b"".join(stream)
        expect_true(bool(data), message="Expected IPC stream payload")

        reader = pa.ipc.open_stream(pa.BufferReader(data))
        metadata = _decode_metadata(reader.schema.metadata or {})
        expect_equal(metadata.get("codeintel.table_key"), expected="docs.v_demo")
        expect_equal(metadata.get("codeintel.repo"), expected=snapshot.repo)
        expect_equal(metadata.get("codeintel.commit"), expected=snapshot.commit)
        expect_equal(metadata.get("codeintel.view_id"), expected="demo.view")
        expect_true(
            bool(metadata.get("codeintel.snapshot_id")),
            message="Expected snapshot_id metadata",
        )
        expect_true(
            bool(metadata.get("codeintel.query_hash")),
            message="Expected query_hash metadata",
        )
        if "codeintel.schema_hash" in metadata:
            expect_true(
                bool(metadata.get("codeintel.schema_hash")),
                message="Expected schema_hash metadata",
            )
        if "codeintel.schema_digest" in metadata:
            expect_true(
                bool(metadata.get("codeintel.schema_digest")),
                message="Expected schema_digest metadata",
            )
        if "codeintel.query_engine" in metadata:
            expect_true(
                bool(metadata.get("codeintel.query_engine")),
                message="Expected query_engine metadata",
            )

        if reader.schema.names != contract_schema.names:
            pytest.fail(f"Schema column mismatch: {reader.schema.names} != {contract_schema.names}")
        if [field.type for field in reader.schema] != [field.type for field in contract_schema]:
            pytest.fail("Schema type mismatch versus contract")
        contract_metadata = _decode_metadata(contract_schema.metadata or {})
        for key, value in contract_metadata.items():
            if metadata.get(key) != value:
                pytest.fail(f"Contract metadata mismatch for {key}: {metadata.get(key)} != {value}")

        row_count = sum(batch.num_rows for batch in reader)
        expect_equal(row_count, expected=2)
    finally:
        await manager.stop()
