"""Tests for FastMCP background task execution (SEP-1686) on the serving surface."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio
import duckdb
import pytest
from fastmcp.client import Client
from mcp import McpError

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.mcp_payloads import extract_payload

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.serving.mcp.protocols import SemanticKernelProtocol as SemanticKernel
    from codeintel.serving.mcp.protocols import ServingDBManagerProtocol
    from codeintel.serving.meta.models import ServingKernelMetaResponse
    from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse
    from codeintel.serving.semantic.models import (
        SemanticCatalogResponse,
        SemanticExplainResponse,
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticQueryResponse,
        SemanticViewDescriptionResponse,
    )


def _make_db(db_path: Path, *, row_count: int) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute(
        "INSERT INTO docs.v_demo SELECT i, 'label-' || i::VARCHAR FROM range(0, ?) t(i)",
        [row_count],
    )
    con.close()


def _write_registry(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_semantic_registry(
        path=path,
        views=[
            {
                "id": "demo.view",
                "kind": "view",
                "table_key": "docs.v_demo",
                "entity": "demo",
                "grain": "per_row",
                "description": "Demo view",
                "primary_key": ["id"],
                "columns": ["id", "label"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
            }
        ],
    )


def _write_schema_manifest(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_schema_manifest(
        path=path,
        tables=[
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
    )


def _write_buildspec(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_buildspec(
        path=path,
        datasets=[{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    )


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    pointer = {
        "db_path": str(db_path),
        "semantic_registry_path": str(registry_path),
        "schema_manifest_path": str(manifest_path),
        "buildspec_path": str(buildspec_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")


def _setup_test_snapshot(tmp_path: Path, *, row_count: int) -> Path:
    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _make_db(db_path, row_count=row_count)
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    return pointer_path


@dataclass(frozen=True, slots=True)
class _SlowExportKernel:
    inner: SemanticKernel
    delay_s: float

    @property
    def db(self) -> ServingDBManagerProtocol:
        return self.inner.db

    def catalog(self) -> SemanticCatalogResponse:
        return self.inner.catalog()

    def describe(self, view_id: str) -> SemanticViewDescriptionResponse:
        return self.inner.describe(view_id)

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        return self.inner.query(request)

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        return self.inner.explain(request)

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        return self.inner.search(request)

    def meta(self) -> ServingKernelMetaResponse:
        return self.inner.meta()

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        for index, row in enumerate(self.inner.export_rows(request)):
            if index > 0:
                time.sleep(self.delay_s)
            yield row

    def export_sql(self, request: SemanticExportRequest) -> str:
        return self.inner.export_sql(request)

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        return self.inner.export_fingerprint(request)

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        return self.inner.export_to_parquet(request, output_path=output_path)

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        return self.inner.export_to_arrow_ipc(request, output_path=output_path)

    def compile_query_sql(self, request: SemanticQueryRequest) -> str:
        return self.inner.compile_query_sql(request)


@pytest.mark.anyio
async def test_mcp_export_task_mode_completes(tmp_path: Path) -> None:
    """Allow calling semantic_export as a background task and awaiting completion."""
    pointer_path = _setup_test_snapshot(tmp_path, row_count=10)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            result_engine="pandas",
            schema_enforcement="strict",
            mcp_mask_errors=False,
            mcp_export_enable_tasks=True,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            task_or_result = await client.call_tool(
                "semantic_export",
                {"request": {"view_id": "demo.view", "export_format": "jsonl", "limit": 10}},
                task=True,
            )

            result_obj = task_or_result
            if hasattr(task_or_result, "result"):
                result_obj = await task_or_result.result()

            payload = extract_payload(result_obj)
            export_id = payload.get("export_id")
            if not isinstance(export_id, str) or not export_id:
                pytest.fail("Expected semantic_export task result to include export_id")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_export_task_cancellation_cleans_up_artifacts(tmp_path: Path) -> None:
    """Cancel a running export task and ensure partial artifacts are cleaned up."""
    pointer_path = _setup_test_snapshot(tmp_path, row_count=2500)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            result_engine="pandas",
            schema_enforcement="strict",
            mcp_mask_errors=False,
            mcp_export_enable_tasks=True,
        )
        inner_kernel = SemanticQueryKernel(db=manager, settings=settings)
        kernel = _SlowExportKernel(inner=inner_kernel, delay_s=0.001)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            tool_task = await client.call_tool(
                "semantic_export",
                {"request": {"view_id": "demo.view", "export_format": "jsonl", "limit": 2500}},
                task=True,
            )
            if not hasattr(tool_task, "cancel"):
                pytest.fail("Expected task-capable client to return a ToolTask")

            await tool_task.cancel()

            exports_dir = tmp_path / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            leftovers: list[str] = []
            for _ in range(200):
                leftovers = [
                    path.name
                    for path in exports_dir.iterdir()
                    if not path.name.endswith(".cancelled")
                ]
                if not leftovers:
                    break
                await anyio.sleep(0.02)
            else:
                pytest.fail(f"Expected export artifacts to be cleaned up, found: {leftovers}")

            with pytest.raises(McpError):
                await tool_task.result()
    finally:
        await manager.stop()
