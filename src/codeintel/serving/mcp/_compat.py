"""FastMCP import shim and feature flags.

This module provides a single import surface for FastMCP, ensuring consistent
usage across the codebase and enabling feature detection for optional capabilities.

All MCP-related code should import from this module rather than directly from
fastmcp to ensure consistent behavior and enable graceful degradation when
optional features are unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Canonical import from gofastmcp 2.x
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import StaticTokenVerifier

if TYPE_CHECKING:
    from fastmcp.server.auth import AuthProvider

LOG = logging.getLogger(__name__)

# Feature detection for EventStore (v2.14.0+)
try:
    from fastmcp.server.event_store import EventStore

    HAS_EVENT_STORE = True
except ImportError:
    EventStore = None  # type: ignore[assignment,misc]
    HAS_EVENT_STORE = False
    LOG.warning("EventStore not available - SSE resumability disabled")


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
