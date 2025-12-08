"""Deprecated shim - use AsyncRecordingMcpRegistrar from tests._helpers.mcp_registrar."""

from __future__ import annotations

from tests._helpers.mcp_registrar import AsyncRecordingMcpRegistrar as AsyncRecordingMcp

__all__ = ["AsyncRecordingMcp"]
