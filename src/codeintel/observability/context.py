"""Correlation context propagation for observability."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.context import Context

try:
    from opentelemetry import baggage as _otel_baggage
    from opentelemetry.context import attach as _otel_attach
    from opentelemetry.context import detach as _otel_detach

    _OTEL_CONTEXT_AVAILABLE = True
except ImportError:
    _OTEL_CONTEXT_AVAILABLE = False
    _otel_baggage = None
    _otel_attach = None
    _otel_detach = None


_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_correlation_id",
    default=None,
)


def get_correlation_id() -> str | None:
    """Return the current correlation identifier.

    Returns
    -------
    str | None
        Correlation identifier when set, otherwise ``None``.
    """
    return _CORRELATION_ID.get()


def set_correlation_id(value: str | None) -> None:
    """Set the current correlation identifier."""
    _CORRELATION_ID.set(value)


@contextlib.contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Set the correlation identifier within a context manager."""
    token = _CORRELATION_ID.set(correlation_id)
    baggage_token: contextvars.Token[Context] | None = None

    if _OTEL_CONTEXT_AVAILABLE and _otel_baggage is not None and _otel_attach is not None:
        baggage = _otel_baggage.set_baggage("correlation_id", correlation_id)
        baggage_token = _otel_attach(baggage)

    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)
        if baggage_token is not None and _otel_detach is not None:
            _otel_detach(baggage_token)


__all__ = ["correlation_context", "get_correlation_id", "set_correlation_id"]
