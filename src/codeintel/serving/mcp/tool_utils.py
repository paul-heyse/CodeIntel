"""Shared MCP tool helpers."""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel
from codeintel.serving.services.query_service import QueryService

if TYPE_CHECKING:
    from collections.abc import Callable

QueryBackendOrService = QueryBackend | QueryService
logger = logging.getLogger(__name__)


def _wrap(func: Callable[..., object]) -> Callable[..., object]:
    """
    Wrap a backend-facing tool to normalize McpError into ProblemDetail payloads.

    Returns
    -------
    Callable[..., object]
        Wrapped tool function that emits dict payloads.
    """

    @wraps(func)
    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except errors.McpError as exc:
            logger.warning("MCP tool error: %s", exc)
            model = ProblemDetailModel.from_domain(exc.detail)
            return {"error": model.model_dump()}

    return _inner


def wrap_tool(func: Callable[..., object]) -> Callable[..., object]:
    """
    Public wrapper that normalizes McpError instances into ProblemDetail payloads.

    Returns
    -------
    Callable[..., object]
        Wrapped function that yields either a result or an error payload.
    """
    return _wrap(func)


__all__ = ["QueryBackendOrService", "_wrap", "wrap_tool"]
