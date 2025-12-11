"""Request-scoped context propagation for serving surfaces."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from contextvars import Token


@dataclass
class RequestContext:
    """
    Per-request context propagated across layers.

    This is transport-agnostic; it captures external (http/mcp/cli)
    and repo-level info. Operation/dataset remain in call metrics.
    """

    correlation_id: str
    transport: Literal["http", "mcp", "cli"]
    operation: str | None = None
    dataset: str | None = None
    repo: str | None = None
    commit: str | None = None
    snapshot: Any | None = None
    graph_scope: Any | None = None
    client_id: str | None = None
    user_agent: str | None = None


_current_request_context: ContextVar[RequestContext | None] = ContextVar(
    "codeintel_current_request_context",
    default=None,
)


def set_current_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    """
    Set the current RequestContext and return a token for reset.

    Returns
    -------
    Token
        ContextVar token to restore the previous RequestContext.
    """
    return _current_request_context.set(ctx)


def get_current_request_context() -> RequestContext | None:
    """
    Return the current RequestContext, if any.

    Returns
    -------
    RequestContext | None
        Active RequestContext when set; otherwise ``None``.
    """
    return _current_request_context.get()


def reset_current_request_context(token: Token[RequestContext | None]) -> None:
    """Reset the current RequestContext to a previous value."""
    _current_request_context.reset(token)


__all__ = [
    "RequestContext",
    "get_current_request_context",
    "reset_current_request_context",
    "set_current_request_context",
]
