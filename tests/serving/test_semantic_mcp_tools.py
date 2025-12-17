"""Tests for semantic MCP tools built from the semantic kernel.

Tests validate that MCP tools return typed Pydantic models (Phase 3 upgrade)
with proper structure and content.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb
import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pool import DuckDBPoolConfig
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

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


def _extract_payload(tool_result: Any) -> dict[str, object]:  # noqa: ANN401
    """Extract the payload from a gofastmcp tool result.

    Parameters
    ----------
    tool_result
        The result from calling a tool via the fastmcp client.
        Can be a CallToolResult object with content attribute,
        or a list of content items, or a plain dict.

    Returns
    -------
    dict[str, object]
        The extracted payload dictionary (full envelope).

    Raises
    ------
    TypeError
        If the result type is unexpected.
    """
    # gofastmcp client.call_tool returns a CallToolResult with content attribute
    if hasattr(tool_result, "content"):
        content_list = tool_result.content
        if content_list and len(content_list) > 0:
            first_content = content_list[0]
            # TextContent has a .text attribute containing JSON
            if hasattr(first_content, "text"):
                return json.loads(first_content.text)

    # Fallback: handle list of content items directly
    if isinstance(tool_result, list) and len(tool_result) > 0:
        first_content = tool_result[0]
        if hasattr(first_content, "text"):
            return json.loads(first_content.text)

    if isinstance(tool_result, dict):
        return tool_result

    msg = f"Unexpected tool result type: {type(tool_result)}"
    raise TypeError(msg)


def _setup_test_snapshot(tmp_path: Path) -> Path:
    """Set up test snapshot files and return pointer path.

    Parameters
    ----------
    tmp_path
        Temporary directory for test files.

    Returns
    -------
    Path
        Path to the pointer JSON file.
    """
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


@pytest.mark.anyio
async def test_mcp_tools_catalog_describe_and_query(tmp_path: Path) -> None:
    """Expose semantic tools over FastMCP and execute them against the snapshot DB.

    Tests now validate typed Pydantic model returns (Phase 3 upgrade):
    - semantic_catalog returns SemanticCatalogResponse
    - semantic_describe returns SemanticViewDescriptionResponse
    - semantic_query returns SemanticQueryToolResponse
    """
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
            mcp_mask_errors=False,  # Disable masking for clearer test errors
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        # Use gofastmcp client pattern for testing
        async with Client(mcp) as client:
            # Test semantic_catalog - returns SemanticCatalogResponse
            catalog_result = _extract_payload(await client.call_tool("semantic_catalog", {}))
            # SemanticCatalogResponse has 'views' list directly
            views_raw = catalog_result.get("views")
            if not isinstance(views_raw, list):
                pytest.fail("Expected semantic_catalog.views to be a list")
            expect_true(any(isinstance(v, dict) and v.get("id") == "demo.view" for v in views_raw))

            # Test semantic_describe - returns SemanticViewDescriptionResponse
            desc_result = _extract_payload(
                await client.call_tool("semantic_describe", {"view_id": "demo.view"})
            )
            # SemanticViewDescriptionResponse has 'table_key' directly
            expect_equal(desc_result.get("table_key"), "docs.v_demo")

            # Test semantic_query - returns SemanticQueryToolResponse
            query_result = _extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {
                        "view_id": "demo.view",
                        "filters": [{"column": "id", "op": "gte", "value": 2}],
                    },
                )
            )
            # SemanticQueryToolResponse has 'result' containing the query response
            result_data = query_result.get("result")
            if not isinstance(result_data, dict):
                pytest.fail("Expected semantic_query.result to be a dict")
            rows_raw = result_data.get("rows")
            if not isinstance(rows_raw, list):
                pytest.fail("Expected semantic_query.result.rows to be a list")
            ids = [row.get("id") for row in rows_raw if isinstance(row, dict)]
            expect_equal(ids, [2, 3])
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_tool_annotations_present(tmp_path: Path) -> None:
    """Verify tools have readOnlyHint annotations."""
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # list_tools() returns list[Tool] directly
            tools = await client.list_tools()

            # Verify all tools have annotations
            for tool in tools:
                expect_true(
                    tool.annotations is not None,
                    message=f"Tool {tool.name} should have annotations",
                )
                # Check readOnlyHint is True for all our tools
                annotations = tool.annotations
                if annotations is not None:
                    expect_true(
                        annotations.readOnlyHint is True,
                        message=f"Tool {tool.name} should have readOnlyHint=True",
                    )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_tool_error_handling(tmp_path: Path) -> None:
    """Verify ToolError returns controlled message for invalid view."""
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "semantic_describe",
                {"view_id": "nonexistent.view"},
                raise_on_error=False,  # Don't raise, check is_error instead
            )
            # Result should indicate an error
            expect_true(result.is_error, message="Expected error for nonexistent view")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_typed_response_structure(tmp_path: Path) -> None:
    """Verify tools return typed Pydantic model structures (Phase 3).

    Tests the structure of typed responses:
    - SemanticCatalogResponse has version, snapshot, views
    - SemanticViewDescriptionResponse has id, table_key, columns, etc.
    - SemanticQueryToolResponse has result (with rows), preview, note
    """
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # Test SemanticCatalogResponse structure
            catalog = _extract_payload(await client.call_tool("semantic_catalog", {}))
            expect_true("version" in catalog, message="Should have version key")
            expect_true("snapshot" in catalog, message="Should have snapshot key")
            expect_true("views" in catalog, message="Should have views key")

            # Validate snapshot structure in catalog
            snapshot = catalog.get("snapshot")
            expect_true(isinstance(snapshot, dict), message="snapshot should be dict")
            if isinstance(snapshot, dict):
                expect_true("repo" in snapshot, message="snapshot should have repo")
                expect_true("commit" in snapshot, message="snapshot should have commit")

            # Test SemanticQueryToolResponse structure
            query = _extract_payload(
                await client.call_tool("semantic_query", {"view_id": "demo.view"})
            )
            expect_true("result" in query, message="Should have result key")
            # preview and note are optional
            result = query.get("result")
            expect_true(isinstance(result, dict), message="result should be dict")
            if isinstance(result, dict):
                expect_true("view_id" in result, message="result should have view_id")
                expect_true("rows" in result, message="result should have rows")
                expect_true("columns" in result, message="result should have columns")
                expect_true("truncated" in result, message="result should have truncated")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_query_response_has_result_data(tmp_path: Path) -> None:
    """Verify SemanticQueryToolResponse contains proper result data.

    With typed responses (Phase 3), the result data is directly in the response,
    no envelope/meta wrapper.
    """
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            response = _extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {"view_id": "demo.view"},
                )
            )

            # SemanticQueryToolResponse has 'result' key
            result = response.get("result")
            expect_is_not_none(result, message="result should be present")
            expect_true(isinstance(result, dict), message="result should be dict")

            if isinstance(result, dict):
                # Result should have rows
                rows = result.get("rows")
                expect_is_not_none(rows, message="rows should be present")
                expect_true(isinstance(rows, list), message="rows should be list")

                # Result should have view_id
                view_id = result.get("view_id")
                expect_equal(view_id, "demo.view")

                # Result should have columns
                columns = result.get("columns")
                expect_is_not_none(columns, message="columns should be present")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_serving_meta_typed_response(tmp_path: Path) -> None:
    """Verify ServingMetaResponse contains correct snapshot values.

    With typed responses (Phase 3), serving_meta returns ServingMetaResponse
    with snapshot, semantic_layer, buildspec, features, limits, etc.
    """
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = _extract_payload(await client.call_tool("serving_meta", {}))

            # ServingMetaResponse has 'snapshot' directly at top level
            snapshot = result.get("snapshot")
            expect_is_not_none(snapshot, message="snapshot should be present")
            expect_true(isinstance(snapshot, dict), message="snapshot should be dict")
            if isinstance(snapshot, dict):
                # These values come from _write_pointer
                expect_equal(snapshot.get("repo"), "demo/repo")
                expect_equal(snapshot.get("commit"), "deadbeef")
                expect_equal(snapshot.get("run_id"), "run-1")

            # ServingMetaResponse has semantic_layer info
            semantic_layer = result.get("semantic_layer")
            expect_is_not_none(semantic_layer, message="semantic_layer should be present")

            # ServingMetaResponse has features dict
            features = result.get("features")
            expect_is_not_none(features, message="features should be present")
            expect_true(isinstance(features, dict), message="features should be dict")

            # ServingMetaResponse has limits
            limits = result.get("limits")
            expect_is_not_none(limits, message="limits should be present")

            # ServingMetaResponse has server_version
            server_version = result.get("server_version")
            expect_is_not_none(server_version, message="server_version should be present")
    finally:
        await manager.stop()
