"""Re-export MCP server configuration for CLI tooling."""

from __future__ import annotations

from codeintel.mcp.config import McpMode, McpServerConfig

__all__ = ["McpMode", "McpServerConfig"]
