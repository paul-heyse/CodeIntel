"""Deprecated shim - use FastMcpAdapter from tests._helpers.mcp_registrar."""

from __future__ import annotations

from tests._helpers.mcp_registrar import FastMcpAdapter as FastMcpRegistrar
from tests._helpers.mcp_registrar import wrap_fastmcp

__all__ = ["FastMcpRegistrar", "wrap_fastmcp"]
