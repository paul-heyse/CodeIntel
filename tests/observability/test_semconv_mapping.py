"""Semantic convention mapping tests."""

from __future__ import annotations

from codeintel.observability.policy import AttributeBudget, ObservabilityPolicy
from codeintel.observability.semconv import http_span_attributes, mcp_span_attributes


def test_http_span_attributes_truncate_route() -> None:
    """HTTP routes should be truncated based on policy limits."""
    policy = ObservabilityPolicy(budget=AttributeBudget(http_route_max_len=5))
    attrs = http_span_attributes(method="GET", route="/abcdefgh", policy=policy)
    assert attrs["http.method"] == "GET"
    assert attrs["http.route"] == "/abc."
    assert set(attrs.keys()) == {"http.method", "http.route"}


def test_mcp_span_attributes_truncate_tool_name() -> None:
    """MCP tool names should be truncated based on policy limits."""
    policy = ObservabilityPolicy(budget=AttributeBudget(mcp_tool_name_max_len=4))
    attrs = mcp_span_attributes(method="tools/call", tool_name="semantic_query", policy=policy)
    assert attrs["mcp.method"] == "tools/call"
    assert attrs["mcp.tool_name"] == "sem."


def test_mcp_span_attributes_handle_missing_tool() -> None:
    """MCP attributes should handle missing tool names."""
    policy = ObservabilityPolicy()
    attrs = mcp_span_attributes(method="tools/list", tool_name=None, policy=policy)
    assert attrs["mcp.method"] == "tools/list"
    assert not attrs["mcp.tool_name"]
