"""Deprecated shim - use RecordingMcpRegistrar from tests._helpers.mcp_registrar."""

from __future__ import annotations

from tests._helpers.mcp_registrar import McpRegistrationRecorder
from tests._helpers.mcp_registrar import RecordingMcpRegistrar as RecordingMcp
from tests._helpers.mcp_registrar import ToolRegistration as McpRegistration

__all__ = ["McpRegistration", "McpRegistrationRecorder", "RecordingMcp"]
