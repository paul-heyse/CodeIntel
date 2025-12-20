"""FastMCP import shim.

This module provides a single import surface for FastMCP, ensuring consistent
usage across the codebase.
"""

from __future__ import annotations

from fastmcp import Context, FastMCP
from fastmcp.server.event_store import EventStore

__all__ = [
    "Context",
    "EventStore",
    "FastMCP",
]
