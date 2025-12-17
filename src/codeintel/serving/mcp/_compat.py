"""FastMCP import shim and feature flags.

This module provides a single import surface for FastMCP, ensuring consistent
usage across the codebase and enabling feature detection for optional capabilities.

All MCP-related code should import from this module rather than directly from
fastmcp to ensure consistent behavior and enable graceful degradation when
optional features are unavailable.
"""

from __future__ import annotations

import logging

# Canonical import from gofastmcp 2.x
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

LOG = logging.getLogger(__name__)

# Feature detection for EventStore (v2.14.0+)
try:
    from fastmcp.server.event_store import EventStore

    HAS_EVENT_STORE = True
except ImportError:
    EventStore = None  # type: ignore[assignment,misc]
    HAS_EVENT_STORE = False
    LOG.warning("EventStore not available - SSE resumability disabled")

__all__ = [
    "HAS_EVENT_STORE",
    "Context",
    "EventStore",
    "FastMCP",
    "ToolError",
]
