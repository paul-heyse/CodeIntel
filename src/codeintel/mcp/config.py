"""Compatibility layer for MCP server configuration."""

from __future__ import annotations

from codeintel.config.serving_models import ServingConfig, ServingMode, verify_db_identity

# Backwards-compatible aliases for existing imports.
McpServerConfig = ServingConfig
McpMode = ServingMode

__all__ = ["McpMode", "McpServerConfig", "ServingConfig", "ServingMode", "verify_db_identity"]
