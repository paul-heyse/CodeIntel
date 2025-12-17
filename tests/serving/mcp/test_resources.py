"""Tests for MCP resources and resource store."""

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
from codeintel.serving.mcp.resource_store import ResourceStore, StoredArtifact
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Test Fixtures Helpers
# ============================================================================


def _make_db(db_path: Path) -> None:
    """Create a test DuckDB database with sample data."""
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.close()


def _write_registry(path: Path) -> None:
    """Write test semantic registry."""
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
    """Write test schema manifest."""
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
    """Write test buildspec."""
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
    """Write test pointer file."""
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
    """Set up test snapshot files.

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


def _extract_payload(tool_result: Any) -> dict[str, object]:  # noqa: ANN401
    """Extract the payload from a gofastmcp tool result.

    Returns
    -------
    dict[str, object]
        The extracted payload dictionary.

    Raises
    ------
    TypeError
        If the result type is unexpected.
    """
    if hasattr(tool_result, "content"):
        content_list = tool_result.content
        if content_list and len(content_list) > 0:
            first_content = content_list[0]
            if hasattr(first_content, "text"):
                return json.loads(first_content.text)

    if isinstance(tool_result, list) and len(tool_result) > 0:
        first_content = tool_result[0]
        if hasattr(first_content, "text"):
            return json.loads(first_content.text)

    if isinstance(tool_result, dict):
        return tool_result

    msg = f"Unexpected tool result type: {type(tool_result)}"
    raise TypeError(msg)


def _extract_data(envelope: dict[str, object]) -> dict[str, object]:
    """Extract the data payload from an envelope.

    Returns
    -------
    dict[str, object]
        The data dictionary from the envelope.

    Raises
    ------
    KeyError
        If data key is missing or not a dict.
    """
    data = envelope.get("data")
    if not isinstance(data, dict):
        msg = "Expected envelope to contain 'data' dict"
        raise KeyError(msg)
    return data


# ============================================================================
# ResourceStore Tests
# ============================================================================


def test_resource_store_put_and_get_json(tmp_path: Path) -> None:
    """Verify JSON artifact storage and retrieval."""
    store = ResourceStore(tmp_path / "exports")

    payload = {"rows": [{"id": 1}, {"id": 2}]}
    token, artifact = store.put_json(payload, row_count=2)

    expect_true(len(token) > 0, message="Token should be non-empty")
    expect_equal(artifact.mime_type, "application/json")
    expect_equal(artifact.row_count, 2)
    expect_true(artifact.size_bytes > 0, message="Size should be positive")
    expect_true(artifact.path.exists(), message="Artifact file should exist")

    # Verify retrieval
    retrieved = store.get(token)
    expect_equal(retrieved.path, artifact.path)
    expect_equal(retrieved.mime_type, "application/json")

    # Verify content
    content = json.loads(artifact.path.read_text(encoding="utf-8"))
    expect_equal(content["rows"], [{"id": 1}, {"id": 2}])


def test_resource_store_put_and_get_ndjson(tmp_path: Path) -> None:
    """Verify NDJSON artifact storage and retrieval."""
    store = ResourceStore(tmp_path / "exports")

    rows = [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}, {"id": 3, "name": "three"}]
    token, artifact = store.put_ndjson(rows)

    expect_true(len(token) > 0, message="Token should be non-empty")
    expect_equal(artifact.mime_type, "application/x-ndjson")
    expect_equal(artifact.row_count, 3)
    expect_true(artifact.size_bytes > 0, message="Size should be positive")
    expect_true(artifact.path.exists(), message="Artifact file should exist")

    # Verify retrieval
    retrieved = store.get(token)
    expect_equal(retrieved.path, artifact.path)
    expect_equal(retrieved.mime_type, "application/x-ndjson")

    # Verify NDJSON content (one JSON per line)
    lines = artifact.path.read_text(encoding="utf-8").strip().split("\n")
    expect_equal(len(lines), 3)
    expect_equal(json.loads(lines[0]), {"id": 1, "name": "one"})
    expect_equal(json.loads(lines[1]), {"id": 2, "name": "two"})
    expect_equal(json.loads(lines[2]), {"id": 3, "name": "three"})


def test_resource_store_get_unknown_token(tmp_path: Path) -> None:
    """Verify KeyError for unknown token."""
    store = ResourceStore(tmp_path / "exports")

    with pytest.raises(KeyError, match="Artifact not found"):
        store.get("nonexistent_token")


def test_resource_store_creates_directory(tmp_path: Path) -> None:
    """Verify ResourceStore creates root directory if it doesn't exist."""
    exports_dir = tmp_path / "nested" / "exports"
    expect_true(not exports_dir.exists(), message="Directory should not exist yet")

    store = ResourceStore(exports_dir)

    expect_true(exports_dir.exists(), message="Directory should be created")
    expect_equal(store.root, exports_dir)


def test_stored_artifact_is_frozen(tmp_path: Path) -> None:
    """Verify StoredArtifact is immutable (frozen dataclass)."""
    artifact = StoredArtifact(
        path=tmp_path / "test.json",
        mime_type="application/json",
        row_count=10,
        size_bytes=100,
    )

    # Frozen dataclass should raise on attribute assignment
    with pytest.raises(AttributeError):
        artifact.row_count = 20  # type: ignore[misc]


# ============================================================================
# MCP Resources Tests
# ============================================================================


@pytest.mark.anyio
async def test_mcp_semantic_export_returns_uri(tmp_path: Path) -> None:
    """Verify semantic_export tool returns a resource URI."""
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
            result = _extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"view_id": "demo.view", "limit": 10},
                )
            )

            # Verify envelope structure
            expect_true("meta" in result, message="Should have meta in envelope")
            expect_true("data" in result, message="Should have data in envelope")

            data = _extract_data(result)

            # Verify export_uri is present
            expect_true("export_uri" in data, message="Should have export_uri")
            uri = data.get("export_uri")
            expect_true(
                isinstance(uri, str) and uri.startswith("codeintel://exports/"),
                message="URI should start with codeintel://exports/",
            )

            # Verify other fields
            expect_equal(data.get("format"), "ndjson")
            expect_equal(data.get("row_count"), 3)  # Demo has 3 rows
            expect_true(
                isinstance(data.get("size_bytes"), int) and data.get("size_bytes", 0) > 0,
                message="size_bytes should be positive",
            )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_json_format(tmp_path: Path) -> None:
    """Verify semantic_export with JSON format."""
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
            result = _extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"view_id": "demo.view", "export_format": "json", "limit": 10},
                )
            )

            data = _extract_data(result)
            expect_equal(data.get("format"), "json")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_with_filters(tmp_path: Path) -> None:
    """Verify semantic_export respects filters."""
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
            result = _extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {
                        "view_id": "demo.view",
                        "filters": [{"column": "id", "op": "eq", "value": 2}],
                    },
                )
            )

            data = _extract_data(result)
            # Filter should return only 1 row (id=2)
            expect_equal(data.get("row_count"), 1)
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_error_invalid_view(tmp_path: Path) -> None:
    """Verify semantic_export returns error for invalid view."""
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
                "semantic_export",
                {"view_id": "nonexistent.view"},
                raise_on_error=False,
            )
            expect_true(result.is_error, message="Expected error for nonexistent view")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_static_resources_available(tmp_path: Path) -> None:
    """Verify static resources are discoverable via list_resources."""
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
            # List all resources
            resources = await client.list_resources()

            # Verify we have our expected resource URIs
            uris = [str(r.uri) for r in resources]

            # Static resources should be present
            expect_true(
                "codeintel://semantic/registry" in uris,
                message="Should have semantic registry resource",
            )
            expect_true(
                "codeintel://meta" in uris,
                message="Should have meta resource",
            )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_read_registry(tmp_path: Path) -> None:
    """Verify reading the semantic registry resource."""
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
            result = await client.read_resource("codeintel://semantic/registry")

            # Result should have content
            expect_true(len(result) > 0, message="Resource should return content")

            # First content item should be the registry data
            content_item = result[0]
            expect_true(hasattr(content_item, "text"), message="Should have text content")

            # Parse and verify structure
            data = json.loads(content_item.text)
            expect_true("version" in data, message="Registry should have version")
            expect_true("views" in data, message="Registry should have views")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_read_meta(tmp_path: Path) -> None:
    """Verify reading the meta resource."""
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
            result = await client.read_resource("codeintel://meta")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = result[0]

            data = json.loads(content_item.text)
            expect_equal(data.get("repo"), "demo/repo")
            expect_equal(data.get("commit"), "deadbeef")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_export_tool_has_annotations(tmp_path: Path) -> None:
    """Verify semantic_export tool has proper annotations."""
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
            tools = await client.list_tools()

            # Find semantic_export tool
            export_tool = next((t for t in tools if t.name == "semantic_export"), None)
            expect_true(export_tool is not None, message="semantic_export tool should exist")

            if export_tool:
                expect_true(
                    export_tool.annotations is not None,
                    message="Tool should have annotations",
                )
                if export_tool.annotations:
                    expect_true(
                        export_tool.annotations.readOnlyHint is True,
                        message="Tool should have readOnlyHint=True",
                    )
    finally:
        await manager.stop()
