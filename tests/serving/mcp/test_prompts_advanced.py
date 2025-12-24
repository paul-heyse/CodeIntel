"""Tests for prompt metadata and elicitation-powered wizard prompts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.serving_snapshots import setup_demo_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import ElicitRequestParams, PromptMessage


_MIN_TOOL_INVOCATIONS = 2


def _setup_test_snapshot(tmp_path: Path) -> Path:
    snapshot = setup_demo_snapshot(tmp_path)
    return snapshot.pointer_path


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
