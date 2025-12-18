"""FastMCP import shim and feature flags.

This module provides a single import surface for FastMCP, ensuring consistent
usage across the codebase and enabling feature detection for optional capabilities.

All MCP-related code should import from this module rather than directly from
fastmcp to ensure consistent behavior and enable graceful degradation when
optional features are unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Canonical import from gofastmcp 2.x
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import StaticTokenVerifier

if TYPE_CHECKING:
    from fastmcp.server.auth import AuthProvider
    from fastmcp.server.event_store import EventStore as FastMCPEventStore

# Feature detection for EventStore (v2.14.0+)
EventStore: type[FastMCPEventStore] | None
try:
    from fastmcp.server.event_store import EventStore as _FastMCPEventStore
except ImportError:
    EventStore = None
    HAS_EVENT_STORE = False
else:
    EventStore = _FastMCPEventStore
    HAS_EVENT_STORE = True


def create_bearer_auth(token: str | None) -> AuthProvider | None:
    """Create a bearer token auth provider if token is provided.

    Parameters
    ----------
    token
        Bearer token string, or None to disable auth.

    Returns
    -------
    AuthProvider | None
        Auth provider for bearer token authentication, or None if no token.
    """
    if not token:
        return None
    # StaticTokenVerifier expects a dict of tokens with optional metadata
    return StaticTokenVerifier({token: {}})


__all__ = [
    "HAS_EVENT_STORE",
    "Context",
    "EventStore",
    "FastMCP",
    "ToolError",
    "create_bearer_auth",
]
