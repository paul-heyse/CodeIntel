"""Security fixtures for serving tests."""

from __future__ import annotations

import secrets
import socket
from ipaddress import IPv4Address

PUBLIC_BIND_HOST = str(IPv4Address(socket.INADDR_ANY))


def auth_token() -> str:
    """Return a test auth token without hardcoding secrets.

    Returns
    -------
    str
        Randomized auth token for tests.
    """
    return secrets.token_urlsafe(16)


def api_key() -> str:
    """Return a test API key without hardcoding secrets.

    Returns
    -------
    str
        Randomized API key for tests.
    """
    return secrets.token_urlsafe(16)


__all__ = ["PUBLIC_BIND_HOST", "api_key", "auth_token"]
