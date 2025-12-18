"""Auth policy shared across HTTP and FastMCP transports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.server.auth import StaticTokenVerifier

from codeintel.serving.errors import AuthForbiddenError
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastmcp.server.auth import AuthProvider


_AUTH_BEARER_PARTS = 2


def configured_tokens(settings: ServingSettings) -> tuple[str, ...]:
    """Return all configured auth tokens (bearer token and/or API key).

    Parameters
    ----------
    settings
        Serving settings providing `auth_token` and `api_key`.

    Returns
    -------
    tuple[str, ...]
        All configured token strings.
    """
    tokens: list[str] = []
    if settings.auth_token:
        tokens.append(settings.auth_token)
    if settings.api_key:
        tokens.append(settings.api_key)
    return tuple(tokens)


def extract_http_token(headers: Mapping[str, str]) -> str | None:
    """Extract an auth token from HTTP headers.

    Supports:
    - ``Authorization: Bearer <token>``
    - ``X-API-Key: <token>``

    Parameters
    ----------
    headers
        HTTP request headers mapping.

    Returns
    -------
    str | None
        Extracted token when present; otherwise None.
    """
    auth = headers.get("Authorization")
    if isinstance(auth, str) and auth:
        parts = auth.split()
        if len(parts) == _AUTH_BEARER_PARTS and parts[0].lower() == "bearer" and parts[1]:
            return parts[1]

    api_key = headers.get("X-API-Key")
    if isinstance(api_key, str) and api_key:
        return api_key

    return None


def require_http_auth(*, headers: Mapping[str, str], settings: ServingSettings) -> None:
    """Enforce auth for HTTP requests when configured.

    Parameters
    ----------
    headers
        HTTP request headers mapping.
    settings
        Serving settings controlling auth requirements.

    Raises
    ------
    AuthForbiddenError
        When credentials are required and missing/invalid.
    """
    expected = configured_tokens(settings)
    if not expected:
        return

    provided = extract_http_token(headers)
    if provided and provided in expected:
        return

    raise AuthForbiddenError(reason="Invalid or missing credentials.")


def mcp_auth_provider(settings: ServingSettings) -> AuthProvider | None:
    """Return an MCP auth provider enforcing the configured tokens.

    Parameters
    ----------
    settings
        Serving settings controlling auth requirements.

    Returns
    -------
    AuthProvider | None
        Auth provider when configured; otherwise None.
    """
    tokens = configured_tokens(settings)
    if not tokens:
        return None
    return StaticTokenVerifier({token: {} for token in tokens})


__all__ = [
    "configured_tokens",
    "extract_http_token",
    "mcp_auth_provider",
    "require_http_auth",
]
