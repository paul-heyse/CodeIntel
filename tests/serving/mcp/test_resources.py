"""Tests for MCP resources and resource store."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.errors import ExportNotFoundError
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.mcp.resource_store import (
    ExportArtifactSpec,
    ResourceStore,
    StoredArtifact,
    StoredMetadata,
)
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshots import setup_demo_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import BlobResourceContents, TextResourceContents

# Expected minimum number of resource templates in canonical taxonomy
_MIN_RESOURCE_TEMPLATES = 8


# ============================================================================
# Test Fixtures Helpers
# ============================================================================


def _setup_test_snapshot(tmp_path: Path) -> Path:
    """Set up test snapshot files.

    Returns
    -------
    Path
        Path to the pointer JSON file.
    """
    snapshot = setup_demo_snapshot(tmp_path)
    return snapshot.pointer_path


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

    rows: list[dict[str, object]] = [{"id": 1}, {"id": 2}]
    token, artifact, _meta = store.put_with_metadata(
        rows,
        spec=ExportArtifactSpec(view_id="demo.view", format="json"),
    )

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


def test_resource_store_put_and_get_jsonl(tmp_path: Path) -> None:
    """Verify JSONL artifact storage and retrieval."""
    store = ResourceStore(tmp_path / "exports")

    rows: list[dict[str, object]] = [
        {"id": 1, "name": "one"},
        {"id": 2, "name": "two"},
        {"id": 3, "name": "three"},
    ]
    token, artifact, _meta = store.put_with_metadata(
        rows,
        spec=ExportArtifactSpec(view_id="demo.view", format="jsonl"),
    )

    expect_true(len(token) > 0, message="Token should be non-empty")
    expect_equal(artifact.mime_type, "application/x-ndjson")
    expect_equal(artifact.row_count, 3)
    expect_true(artifact.size_bytes > 0, message="Size should be positive")
    expect_true(artifact.path.exists(), message="Artifact file should exist")

    # Verify retrieval
    retrieved = store.get(token)
    expect_equal(retrieved.path, artifact.path)
    expect_equal(retrieved.mime_type, "application/x-ndjson")

    # Verify JSONL content (one JSON per line)
    lines = artifact.path.read_text(encoding="utf-8").strip().split("\n")
    expect_equal(len(lines), 3)
    expect_equal(json.loads(lines[0]), {"id": 1, "name": "one"})
    expect_equal(json.loads(lines[1]), {"id": 2, "name": "two"})
    expect_equal(json.loads(lines[2]), {"id": 3, "name": "three"})


def test_resource_store_jsonl_encoding_is_utf8(tmp_path: Path) -> None:
    """Verify JSONL encoding preserves UTF-8 and stringifies unknown types."""
    store = ResourceStore(tmp_path / "exports")

    expected_ts = "2024-01-01T00:00:00Z"
    row: dict[str, object] = {
        "text": "naïve 🧪",
        "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "bytes": b"hello",
    }
    token, artifact, _meta = store.put_with_metadata(
        [row],
        spec=ExportArtifactSpec(view_id="demo.view", format="jsonl"),
    )

    expect_true(len(token) > 0, message="Token should be non-empty")

    raw = artifact.path.read_text(encoding="utf-8").strip()
    expect_true(bool(raw), message="Expected JSONL output")
    expect_equal(raw.count("\n"), 0)
    expect_true("🧪" in raw, message="Expected UTF-8 output to preserve unicode")

    payload = json.loads(raw)
    expect_equal(payload["text"], row["text"])
    expect_equal(payload["ts"], expected_ts)
    expect_equal(payload["uuid"], str(row["uuid"]))
    expect_equal(payload["bytes"], str(row["bytes"]))


def test_resource_store_jsonl_stream_encoding_is_utf8(tmp_path: Path) -> None:
    """Verify streaming JSONL encoding preserves UTF-8 and stringifies types."""
    store = ResourceStore(tmp_path / "exports")

    expected_ts = "2024-01-01T00:00:00Z"
    row: dict[str, object] = {
        "text": "naïve 🧪",
        "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "bytes": b"hello",
    }

    token, artifact, _meta = store.put_with_metadata_stream(
        [row],
        spec=ExportArtifactSpec(view_id="demo.view", format="jsonl"),
    )

    expect_true(len(token) > 0, message="Token should be non-empty")

    raw = artifact.path.read_text(encoding="utf-8").strip()
    expect_true(bool(raw), message="Expected JSONL output")
    expect_equal(raw.count("\n"), 0)
    expect_true("🧪" in raw, message="Expected UTF-8 output to preserve unicode")

    payload = json.loads(raw)
    expect_equal(payload["text"], row["text"])
    expect_equal(payload["ts"], expected_ts)
    expect_equal(payload["uuid"], str(row["uuid"]))
    expect_equal(payload["bytes"], str(row["bytes"]))


def test_resource_store_get_unknown_token(tmp_path: Path) -> None:
    """Verify ExportNotFoundError for unknown token."""
    store = ResourceStore(tmp_path / "exports")

    with pytest.raises(ExportNotFoundError):
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


def test_resource_store_put_with_metadata(tmp_path: Path) -> None:
    """Verify put_with_metadata stores both artifact and metadata sidecar."""
    store = ResourceStore(tmp_path / "exports")

    rows: list[dict[str, object]] = [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}]
    columns = ("id", "name")
    column_types = {"id": "INTEGER", "name": "VARCHAR"}
    snapshot = {"repo": "demo/repo", "commit": "abc123", "run_id": "run-1"}

    spec = ExportArtifactSpec(
        view_id="demo.view",
        columns=columns,
        column_types=column_types,
        compiled_sql="SELECT * FROM demo",
        snapshot=snapshot,
        format="jsonl",
    )
    token, artifact, metadata = store.put_with_metadata(rows, spec=spec)

    # Verify token and artifact
    expect_true(len(token) > 0, message="Token should be non-empty")
    expect_equal(artifact.mime_type, "application/x-ndjson")
    expect_equal(artifact.row_count, 2)

    # Verify metadata object
    expect_equal(metadata.export_id, token)
    expect_equal(metadata.view_id, "demo.view")
    expect_equal(metadata.row_count, 2)
    expect_equal(metadata.columns, columns)
    expect_equal(metadata.column_types, column_types)
    expect_equal(metadata.compiled_sql, "SELECT * FROM demo")
    expect_equal(metadata.snapshot["repo"], "demo/repo")

    # Verify sidecar file exists
    meta_path = tmp_path / "exports" / f"{token}.meta.json"
    expect_true(meta_path.exists(), message="Metadata sidecar should exist")


def test_resource_store_get_meta(tmp_path: Path) -> None:
    """Verify get_meta retrieves stored metadata."""
    store = ResourceStore(tmp_path / "exports")

    rows: list[dict[str, object]] = [{"id": 1}]
    spec = ExportArtifactSpec(
        view_id="test.view",
        columns=("id",),
        compiled_sql="SELECT id FROM test",
        snapshot={"repo": "test"},
        format="jsonl",
    )
    token, _artifact, _metadata = store.put_with_metadata(rows, spec=spec)

    # Retrieve metadata
    retrieved = store.get_meta(token)
    expect_equal(retrieved.export_id, token)
    expect_equal(retrieved.view_id, "test.view")
    expect_equal(retrieved.columns, ("id",))
    expect_equal(retrieved.compiled_sql, "SELECT id FROM test")


def test_resource_store_get_meta_not_found(tmp_path: Path) -> None:
    """Verify get_meta raises ExportNotFoundError for unknown token."""
    store = ResourceStore(tmp_path / "exports")

    with pytest.raises(ExportNotFoundError):
        store.get_meta("nonexistent_token")


def test_resource_store_get_preview(tmp_path: Path) -> None:
    """Verify get_preview returns first N rows."""
    store = ResourceStore(tmp_path / "exports")

    rows: list[dict[str, object]] = [{"id": i, "name": f"item-{i}"} for i in range(10)]
    spec = ExportArtifactSpec(view_id="test.view", columns=("id", "name"), format="jsonl")
    token, _artifact, _metadata = store.put_with_metadata(rows, spec=spec)

    # Get preview (default 5 rows)
    preview = store.get_preview(token, max_rows=5)
    expect_equal(preview["export_id"], token)
    expect_equal(preview["preview_row_count"], 5)
    expect_equal(preview["total_row_count"], 10)
    expect_true(preview["truncated"] is True, message="Should be truncated")
    expect_equal(len(cast("list[object]", preview["rows"])), 5)


def test_resource_store_get_preview_small_dataset(tmp_path: Path) -> None:
    """Verify get_preview handles small datasets correctly."""
    store = ResourceStore(tmp_path / "exports")

    rows: list[dict[str, object]] = [{"id": 1}, {"id": 2}]
    spec = ExportArtifactSpec(view_id="test.view", columns=("id",), format="jsonl")
    token, _artifact, _metadata = store.put_with_metadata(rows, spec=spec)

    preview = store.get_preview(token, max_rows=5)
    expect_equal(preview["preview_row_count"], 2)
    expect_equal(preview["total_row_count"], 2)
    expect_true(preview["truncated"] is False, message="Should not be truncated")


def test_stored_metadata_is_frozen() -> None:
    """Verify StoredMetadata is immutable (frozen dataclass)."""
    metadata = StoredMetadata(
        export_id="token123",
        view_id="test.view",
        row_count=10,
        columns=("id",),
    )

    with pytest.raises(AttributeError):
        metadata.row_count = 20  # type: ignore[misc]


# ============================================================================
# MCP Resources Tests
# ============================================================================


@pytest.mark.anyio
async def test_mcp_semantic_export_returns_uri(tmp_path: Path) -> None:
    """Verify semantic_export tool returns a resource URI."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "limit": 10}},
                )
            )

            # Result is now typed (ExportHandleResponse), verify fields directly
            expect_true("export_id" in result, message="Should have export_id")
            expect_true("uri" in result, message="Should have uri")
            uri = result.get("uri")
            expect_true(
                isinstance(uri, str) and uri.startswith("codeintel://exports/"),
                message="URI should start with codeintel://exports/",
            )

            # Verify other fields
            expect_equal(result.get("format"), "jsonl")
            expect_equal(result.get("row_count"), 3)  # Demo has 3 rows
            byte_size = result.get("byte_size", 0)
            expect_true(
                isinstance(byte_size, int) and byte_size > 0,
                message="byte_size should be positive",
            )

            # Verify meta_uri for sub-resources
            meta_uri = result.get("meta_uri")
            expect_true(
                isinstance(meta_uri, str) and "/meta" in meta_uri,
                message="meta_uri should point to metadata resource",
            )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_json_format(tmp_path: Path) -> None:
    """Verify semantic_export with JSON format."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "export_format": "json", "limit": 10}},
                )
            )

            # Result is now typed (ExportHandleResponse)
            expect_equal(result.get("format"), "json")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_with_filters(tmp_path: Path) -> None:
    """Verify semantic_export respects filters."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {
                        "request": {
                            "view_id": "demo.view",
                            "filters": [{"column": "id", "op": "eq", "value": 2}],
                        },
                    },
                )
            )

            # Result is now typed (ExportHandleResponse)
            # Filter should return only 1 row (id=2)
            expect_equal(result.get("row_count"), 1)
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_semantic_export_error_invalid_view(tmp_path: Path) -> None:
    """Verify semantic_export returns error for invalid view."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "semantic_export",
                {"request": {"view_id": "nonexistent.view"}},
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # List all resources
            resources = await client.list_resources()

            # Verify we have our expected resource URIs (canonical taxonomy)
            uris = [str(r.uri) for r in resources]

            # Static resources with canonical URIs should be present
            expect_true(
                "codeintel://semantic/views" in uris,
                message="Should have semantic views resource",
            )
            expect_true(
                "codeintel://meta/serving" in uris,
                message="Should have meta/serving resource",
            )
            expect_true(
                "codeintel://meta/resources" in uris,
                message="Should have meta/resources discovery resource",
            )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_read_semantic_views(tmp_path: Path) -> None:
    """Verify reading the semantic views resource (canonical URI)."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # Use canonical URI: codeintel://semantic/views
            result = await client.read_resource("codeintel://semantic/views")

            # Result should have content
            expect_true(len(result) > 0, message="Resource should return content")

            # First content item should be the registry data
            content_item = cast("TextResourceContents", result[0])
            expect_true(hasattr(content_item, "text"), message="Should have text content")

            # Parse and verify structure
            data = json.loads(content_item.text)
            expect_true("version" in data, message="Registry should have version")
            expect_true("views" in data, message="Registry should have views")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_read_meta_serving(tmp_path: Path) -> None:
    """Verify reading the meta/serving resource (canonical URI)."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # Use canonical URI: codeintel://meta/serving
            result = await client.read_resource("codeintel://meta/serving")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = cast("TextResourceContents", result[0])

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


@pytest.mark.anyio
async def test_mcp_resource_meta_resources(tmp_path: Path) -> None:
    """Verify codeintel://meta/resources returns discovery catalog."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = await client.read_resource("codeintel://meta/resources")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = cast("TextResourceContents", result[0])

            data = json.loads(content_item.text)

            # Verify ResourceTemplatesResponse structure
            expect_equal(data.get("uri"), "codeintel://meta/resources")
            expect_true("generated_at" in data, message="Should have generated_at")
            expect_true("snapshot" in data, message="Should have snapshot")
            expect_true("templates" in data, message="Should have templates")

            # Verify templates list
            templates = data.get("templates", [])
            expect_true(
                len(templates) >= _MIN_RESOURCE_TEMPLATES,
                message=f"Should have at least {_MIN_RESOURCE_TEMPLATES} templates",
            )

            # Check some specific templates exist
            template_uris = [t.get("uri") for t in templates]
            expect_true(
                "codeintel://meta/serving" in template_uris,
                message="Should have meta/serving template",
            )
            expect_true(
                "codeintel://semantic/views" in template_uris,
                message="Should have semantic/views template",
            )
            expect_true(
                "codeintel://exports/{export_id}/meta" in template_uris,
                message="Should have exports meta template",
            )
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_meta_environment(tmp_path: Path) -> None:
    """Verify codeintel://meta/environment returns environment + runtime info."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            result = await client.read_resource("codeintel://meta/environment")
            content_item = cast("TextResourceContents", result[0])
            data = json.loads(content_item.text)
            expect_true("mcp_export_limits" in data, message="Should include export limits")
            expect_true("runtime_versions" in data, message="Should include runtime_versions")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_meta(tmp_path: Path) -> None:
    """Verify codeintel://exports/{export_id}/meta returns export metadata."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # First create an export
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "limit": 10}},
                )
            )
            export_id = export_result.get("export_id")
            expect_true(export_id is not None, message="Should have export_id")

            # Now read the meta resource
            result = await client.read_resource(f"codeintel://exports/{export_id}/meta")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = cast("TextResourceContents", result[0])

            data = json.loads(content_item.text)

            # Verify ExportMetaResponse structure
            expect_equal(data.get("export_id"), export_id)
            expect_equal(data.get("status"), "ready")
            expect_true("created_at" in data, message="Should have created_at")
            expect_equal(data.get("format"), "jsonl")
            expect_equal(data.get("row_count"), 3)  # Demo view has 3 rows

            # Verify snapshot info
            expect_true("snapshot" in data, message="Should have snapshot")
            snapshot = data.get("snapshot", {})
            expect_true("snapshot" in snapshot, message="Snapshot should have nested snapshot")

            # Verify URIs
            expect_true("uris" in data, message="Should have uris")
            uris = data.get("uris", {})
            expect_true("payload_uri" in uris, message="Should have payload_uri")
            expect_true("meta_uri" in uris, message="Should have meta_uri")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_preview(tmp_path: Path) -> None:
    """Verify codeintel://exports/{export_id}/preview returns LLM-friendly preview."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # First create an export
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "limit": 10}},
                )
            )
            export_id = export_result.get("export_id")

            # Now read the preview resource
            result = await client.read_resource(f"codeintel://exports/{export_id}/preview")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = cast("TextResourceContents", result[0])

            data = json.loads(content_item.text)

            # Verify preview structure
            expect_equal(data.get("export_id"), export_id)
            expect_true("columns" in data, message="Should have columns")
            expect_true("rows" in data, message="Should have rows")
            expect_true("preview_row_count" in data, message="Should have preview_row_count")
            expect_true("total_row_count" in data, message="Should have total_row_count")

            # Demo view has 3 rows, preview should show all (< 5 max)
            expect_equal(data.get("total_row_count"), 3)
            expect_equal(data.get("preview_row_count"), 3)
            expect_true(data.get("truncated") is False, message="Should not be truncated")

            # Verify actual row content
            rows = data.get("rows", [])
            expect_equal(len(rows), 3)
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_lines_chunk(tmp_path: Path) -> None:
    """Verify codeintel://exports/{export_id}/lines supports chunked line reads."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "limit": 10}},
                )
            )
            export_id = export_result["export_id"]

            chunk = await client.read_resource(
                f"codeintel://exports/{export_id}/lines?offset=0&limit=2"
            )
            content_item = cast("TextResourceContents", chunk[0])
            lines = [line for line in content_item.text.splitlines() if line.strip()]
            expect_equal(len(lines), 2)
            _ = json.loads(lines[0])
            _ = json.loads(lines[1])
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_meta_row_count_for_parquet(tmp_path: Path) -> None:
    """Verify parquet exports record row_count and expose binary URIs."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {
                        "request": {
                            "view_id": "demo.view",
                            "export_format": "parquet",
                            "limit": 10,
                        }
                    },
                )
            )
            export_id = export_result["export_id"]

            meta_result = await client.read_resource(f"codeintel://exports/{export_id}/meta")
            content_item = cast("TextResourceContents", meta_result[0])
            meta = json.loads(content_item.text)

            expect_equal(meta.get("format"), "parquet")
            expect_equal(meta.get("row_count"), 3)
            uris = meta.get("uris", {})
            expect_true(isinstance(uris, dict))
            expect_true(uris.get("bytes_uri_template") is not None)
            expect_equal(uris.get("lines_uri_template"), None)
            expect_equal(uris.get("preview_uri"), None)
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_bytes_chunk_for_arrow(tmp_path: Path) -> None:
    """Verify codeintel://exports/{export_id}/bytes supports chunked bytes reads for arrow exports."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "export_format": "arrow", "limit": 10}},
                )
            )
            export_id = export_result["export_id"]

            chunk = await client.read_resource(
                f"codeintel://exports/{export_id}/bytes?offset=0&limit=128"
            )
            blob_item = cast("BlobResourceContents", chunk[0])
            payload = base64.b64decode(blob_item.blob)
            expect_true(len(payload) > 0, message="Should return some bytes")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_resource_export_sql(tmp_path: Path) -> None:
    """Verify codeintel://exports/{export_id}/sql returns SQL or placeholder."""
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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            # First create an export
            export_result = extract_payload(
                await client.call_tool(
                    "semantic_export",
                    {"request": {"view_id": "demo.view", "limit": 10}},
                )
            )
            export_id = export_result.get("export_id")

            # Now read the SQL resource
            result = await client.read_resource(f"codeintel://exports/{export_id}/sql")

            expect_true(len(result) > 0, message="Resource should return content")
            content_item = cast("TextResourceContents", result[0])

            # SQL may be placeholder if not recorded during export
            sql_text = content_item.text
            expect_true(len(sql_text) > 0, message="Should return non-empty SQL or placeholder")
            expect_true(
                sql_text.startswith(("--", "SELECT")),
                message="Should be SQL or comment placeholder",
            )
    finally:
        await manager.stop()
