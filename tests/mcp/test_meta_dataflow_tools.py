"""MCP smoke tests for dataflow introspection tools."""

from __future__ import annotations

import pytest

from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.storage.metadata import bootstrap_metadata_datasets
from tests._helpers.gateway import GatewayFactory, build_duckdb_backend


def test_explain_dataset_tool_smoke() -> None:
    """explain_dataset should return a payload for known dataset nodes."""
    fastmcp_mod = pytest.importorskip("mcp.server.fastmcp")
    mcp = fastmcp_mod.FastMCP("test")
    if not hasattr(mcp, "tools"):
        pytest.skip("FastMCP tools registry not available")

    gateway = GatewayFactory().without_validation().open()
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        backend = build_duckdb_backend(gateway, repo="demo/repo", commit="deadbeef")

        register_meta_tools(mcp, backend)
        explain = mcp.tools.get("explain_dataset")
        if explain is None:
            pytest.fail("explain_dataset tool not registered")

        result = explain({"node_id": "analytics.function_profile"})
        if isinstance(result, dict) and "error" in result:
            pytest.fail(f"Unexpected error payload from explain_dataset: {result}")
        if not isinstance(result, list):
            pytest.fail(f"Unexpected explain_dataset payload type: {type(result)}")
        if not result:
            pytest.fail("Expected explain_dataset to return a payload")
    finally:
        gateway.close()
