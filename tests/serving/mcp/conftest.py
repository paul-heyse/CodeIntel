"""Legacy MCP fixtures re-exported for serving tests.

Pyrefly and downstream tests expect these symbols at this path; we forward
to the shared plugin implementations to avoid duplication.
"""

from __future__ import annotations

from tests._helpers.plugins.mcp import (
    McpBackendComponents,
    mcp_backend,
    mcp_backend_components,
    mcp_backend_factory,
    mcp_query_service,
    mcp_service,
)

__all__ = [
    "McpBackendComponents",
    "mcp_backend",
    "mcp_backend_components",
    "mcp_backend_factory",
    "mcp_query_service",
    "mcp_service",
]
