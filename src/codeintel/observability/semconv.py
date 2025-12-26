"""Semantic convention helpers for observability attributes."""

from __future__ import annotations

from codeintel.observability.attributes import SpanAttributeValue, shape_attributes
from codeintel.observability.policy import ObservabilityPolicy


def http_span_attributes(
    *,
    method: str,
    route: str,
    policy: ObservabilityPolicy,
) -> dict[str, SpanAttributeValue]:
    """Return normalized HTTP span attributes.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Normalized HTTP span attributes.
    """
    normalized_route = _truncate(route, policy.http_route_max_len)
    attrs = {
        "http.method": method,
        "http.route": normalized_route,
    }
    return shape_attributes(attrs, allowed_keys=frozenset(attrs.keys()))


def mcp_span_attributes(
    *,
    method: str,
    tool_name: str | None,
    policy: ObservabilityPolicy,
) -> dict[str, SpanAttributeValue]:
    """Return normalized MCP span attributes.

    Returns
    -------
    dict[str, SpanAttributeValue]
        Normalized MCP span attributes.
    """
    normalized_tool = None
    if tool_name:
        normalized_tool = _truncate(tool_name, policy.mcp_tool_name_max_len)
    attrs = {
        "mcp.method": method,
        "mcp.tool_name": normalized_tool or "",
    }
    return shape_attributes(attrs, allowed_keys=frozenset(attrs.keys()))


def _truncate(value: str, max_len: int) -> str:
    if max_len < 0:
        return value
    if len(value) <= max_len:
        return value
    if max_len == 0:
        return ""
    if max_len == 1:
        return value[:1]
    return value[: max_len - 1] + "."


__all__ = ["http_span_attributes", "mcp_span_attributes"]
