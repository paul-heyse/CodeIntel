"""Shared MCP tool helpers."""

from __future__ import annotations

from collections.abc import Callable

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.services.query_service import QueryService

QueryBackendOrService = QueryBackend | QueryService


def _wrap(func: Callable[..., object]) -> Callable[..., object]:
    """
    Wrap a backend-facing tool to normalize McpError into ProblemDetail payloads.

    Returns
    -------
    Callable[..., object]
        Wrapped tool function that emits dict payloads.
    """

    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except errors.McpError as exc:
            return {"error": exc.detail.model_dump()}

    return _inner


__all__ = ["QueryBackendOrService", "_wrap"]
