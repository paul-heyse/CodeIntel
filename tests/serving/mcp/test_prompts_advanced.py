"""Tests for prompt metadata and elicitation-powered wizard prompts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
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

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import ElicitRequestParams, PromptMessage


_MIN_TOOL_INVOCATIONS = 2


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
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


def _message_text(message: PromptMessage) -> str:
    content = message.content
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return ""


@pytest.mark.anyio
async def test_mcp_list_prompts_includes_tags_and_version_meta(tmp_path: Path) -> None:
    """Expose prompt version meta and FastMCP tags via list_prompts."""
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
            prompts = await client.list_prompts()
            prompt = next((p for p in prompts if p.name == "wizard_query_view"), None)
            if prompt is None:
                pytest.fail("Expected wizard_query_view to be registered")

            if not isinstance(prompt.meta, dict):
                pytest.fail("Expected prompt meta to be a dict")
            if prompt.meta.get("version") != 1:
                pytest.fail("Expected wizard_query_view meta.version == 1")

            fastmcp_meta = prompt.meta.get("_fastmcp")
            if not isinstance(fastmcp_meta, dict):
                pytest.fail("Expected prompt meta to include _fastmcp")
            tags = fastmcp_meta.get("tags")
            if not isinstance(tags, list) or not tags:
                pytest.fail("Expected prompt meta._fastmcp.tags to be a non-empty list")
            if "wizard" not in tags:
                pytest.fail("Expected prompt tags to include 'wizard'")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_get_prompt_wizard_query_view_uses_elicitation(tmp_path: Path) -> None:
    """Use elicitation to produce ready-to-run tool invocations from wizard prompt."""
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

        responses = iter(
            [
                {"value": "demo.view"},
                {"value": ""},  # select all columns
                {"value": "no"},  # no filters
            ]
        )

        async def elicitation_handler(
            _message: str,
            _response_type: type[object] | None,
            _params: ElicitRequestParams,
            _context: object,
        ) -> dict[str, Any]:
            await anyio.sleep(0.0001)
            return next(responses)

        async with Client(mcp, elicitation_handler=elicitation_handler) as client:
            prompt_result = await client.get_prompt("wizard_query_view")
            messages = prompt_result.messages
            texts = [_message_text(m) for m in messages]

            invocation_texts = [
                text for text in texts if '"tool"' in text and '"arguments"' in text
            ]
            if len(invocation_texts) < _MIN_TOOL_INVOCATIONS:
                pytest.fail("Expected wizard prompt to include tool invocation JSON messages")

            query_call = next((t for t in invocation_texts if '"semantic_query"' in t), None)
            if query_call is None:
                pytest.fail("Expected wizard prompt to include semantic_query invocation")
            parsed = json.loads(query_call)
            args = parsed.get("arguments")
            if not isinstance(args, dict):
                pytest.fail("Expected tool invocation to contain arguments dict")
            request = args.get("request")
            if not isinstance(request, dict):
                pytest.fail("Expected tool invocation to contain request dict")
            if request.get("view_id") != "demo.view":
                pytest.fail("Expected wizard semantic_query invocation to target demo.view")
    finally:
        await manager.stop()
