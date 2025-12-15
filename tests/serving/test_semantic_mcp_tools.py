"""Tests for semantic MCP tools built from the semantic kernel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pool import DuckDBPoolConfig
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from pathlib import Path


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.close()


def _write_registry(path: Path) -> None:
    registry = {
        "version": "v1",
        "views": [
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
    }
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def _write_schema_manifest(path: Path) -> None:
    manifest = {
        "version": "v1",
        "tables": [
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
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_buildspec(path: Path) -> None:
    buildspec = {
        "spec_version": 1,
        "targets": [],
        "datasets": [{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    }
    path.write_text(json.dumps(buildspec, indent=2, sort_keys=True), encoding="utf-8")


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


def _extract_payload(tool_result: object) -> dict[str, object]:
    if isinstance(tool_result, tuple):
        match tool_result:
            case (_content, payload) if isinstance(payload, dict):
                return payload
            case (_content, payload):
                msg = f"Unexpected payload type: {type(payload)}"
                raise TypeError(msg)
            case _:
                msg = f"Unexpected tool result tuple: {tool_result!r}"
                raise TypeError(msg)

    if isinstance(tool_result, dict):
        return tool_result

    msg = f"Unexpected tool result type: {type(tool_result)}"
    raise TypeError(msg)


@pytest.mark.anyio
async def test_mcp_tools_catalog_describe_and_query(tmp_path: Path) -> None:
    """Expose semantic tools over FastMCP and execute them against the snapshot DB."""
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

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(db=manager)
        mcp = build_mcp_app(kernel=kernel, streamable_http_path="/mcp")

        catalog = _extract_payload(await mcp.call_tool("semantic_catalog", {}))
        views_raw = catalog.get("views")
        if not isinstance(views_raw, list):
            pytest.fail("Expected semantic_catalog.views to be a list")
        expect_true(any(isinstance(v, dict) and v.get("id") == "demo.view" for v in views_raw))

        desc = _extract_payload(await mcp.call_tool("semantic_describe", {"view_id": "demo.view"}))
        expect_equal(desc.get("table_key"), "docs.v_demo")

        query = _extract_payload(
            await mcp.call_tool(
                "semantic_query",
                {
                    "view_id": "demo.view",
                    "filters": [{"column": "id", "op": "gte", "value": 2}],
                },
            )
        )
        rows_raw = query.get("rows")
        if not isinstance(rows_raw, list):
            pytest.fail("Expected semantic_query.rows to be a list")
        ids = [row.get("id") for row in rows_raw if isinstance(row, dict)]
        expect_equal(ids, [2, 3])
    finally:
        await manager.stop()
