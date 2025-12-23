"""Tests for FastMCP sampling integration (ctx.sample) on the serving surface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import TYPE_CHECKING

import duckdb
import pytest
from fastmcp.client import Client

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.mcp_payloads import extract_payload

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import CreateMessageRequestParams, SamplingMessage


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo SELECT i, 'label-' || i::VARCHAR FROM range(0, 30) t(i)")
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
                "deprecated": False,
                "replaced_by": None,
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


def _setup_test_snapshot(tmp_path: Path) -> Path:
    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _make_db(db_path)
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


def _runtime_version(name: str) -> str:
    try:
        return get_package_version(name)
    except PackageNotFoundError:
        return "not-installed"


@pytest.mark.anyio
async def test_mcp_sampling_opt_in_adds_summary(tmp_path: Path) -> None:
    """Include a summary only when sampling is enabled and supported."""
    pointer_path = _setup_test_snapshot(tmp_path)

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
            mcp_enable_sampling=True,
            mcp_sample_threshold=1,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        def sampling_handler(
            _messages: list[SamplingMessage],
            _params: CreateMessageRequestParams,
            _context: object,
        ) -> str:
            return f"summary(runtime_sqlglot={_runtime_version('sqlglot')})"

        async with Client(mcp, sampling_handler=sampling_handler) as client:
            payload = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {"request": {"view_id": "demo.view"}},
                )
            )
            summary = payload.get("summary")
            if not isinstance(summary, str) or "summary(" not in summary:
                pytest.fail("Expected semantic_query to include sampling summary when enabled")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_sampling_disabled_does_not_sample(tmp_path: Path) -> None:
    """Avoid calling ctx.sample when server-side sampling is disabled."""
    pointer_path = _setup_test_snapshot(tmp_path)

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
            mcp_enable_sampling=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        def sampling_handler(
            _messages: list[SamplingMessage],
            _params: CreateMessageRequestParams,
            _context: object,
        ) -> str:
            return "summary(should_not_be_used)"

        async with Client(mcp, sampling_handler=sampling_handler) as client:
            payload = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {"request": {"view_id": "demo.view"}},
                )
            )
            if payload.get("summary") is not None:
                pytest.fail("Expected semantic_query to omit summary when sampling is disabled")
    finally:
        await manager.stop()
