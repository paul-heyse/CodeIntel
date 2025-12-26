"""Semantic convention helpers for observability attributes."""

from __future__ import annotations

from codeintel.observability.attribute_sanitizer import (
    SpanAttributeValue,
    shape_attributes,
    truncate_str,
)
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.semconv_keys import (
    HTTP_METHOD,
    HTTP_ROUTE,
    MCP_METHOD,
    MCP_TOOL_NAME,
)


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
    normalized_route = truncate_str(route, policy.budget.http_route_max_len)
    attrs = {
        HTTP_METHOD: method,
        HTTP_ROUTE: normalized_route,
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
        normalized_tool = truncate_str(tool_name, policy.budget.mcp_tool_name_max_len)
    attrs = {
        MCP_METHOD: method,
        MCP_TOOL_NAME: normalized_tool or "",
    }
    return shape_attributes(attrs, allowed_keys=frozenset(attrs.keys()))


__all__ = ["http_span_attributes", "mcp_span_attributes"]
