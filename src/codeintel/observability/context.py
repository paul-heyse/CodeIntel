"""Correlation context propagation for observability."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from opentelemetry import baggage as _otel_baggage
from opentelemetry.context import Context
from opentelemetry.context import attach as _otel_attach
from opentelemetry.context import detach as _otel_detach

_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_correlation_id",
    default=None,
)
_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_run_id",
    default=None,
)
_DOMAIN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_domain",
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


def get_run_id() -> str | None:
    """Return the current run identifier.

    Returns
    -------
    str | None
        Run identifier when set, otherwise ``None``.
    """
    return _RUN_ID.get()


def set_run_id(value: str | None) -> None:
    """Set the current run identifier."""
    _RUN_ID.set(value)


def get_domain() -> str | None:
    """Return the current domain identifier.

    Returns
    -------
    str | None
        Domain identifier when set, otherwise ``None``.
    """
    return _DOMAIN.get()


def set_domain(value: str | None) -> None:
    """Set the current domain identifier."""
    _DOMAIN.set(value)


@contextlib.contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Set the correlation identifier within a context manager."""
    token = _CORRELATION_ID.set(correlation_id)
    baggage_token: contextvars.Token[Context] | None = None

    baggage = _otel_baggage.set_baggage("correlation_id", correlation_id)
    baggage_token = _otel_attach(baggage)

    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)
        if baggage_token is not None:
            _otel_detach(baggage_token)


@contextlib.contextmanager
def run_context(*, run_id: str | None = None, domain: str | None = None) -> Iterator[None]:
    """Set run and domain identifiers within a context manager."""
    run_token = _RUN_ID.set(run_id)
    domain_token = _DOMAIN.set(domain)
    baggage_token: contextvars.Token[Context] | None = None

    baggage = None
    if run_id:
        baggage = _otel_baggage.set_baggage("codeintel.run_id", run_id)
    if domain:
        baggage = _otel_baggage.set_baggage(
            "codeintel.domain",
            domain,
            baggage,
        )
    if baggage is not None:
        baggage_token = _otel_attach(baggage)

    try:
        yield
    finally:
        _RUN_ID.reset(run_token)
        _DOMAIN.reset(domain_token)
        if baggage_token is not None:
            _otel_detach(baggage_token)


__all__ = [
    "correlation_context",
    "get_correlation_id",
    "get_domain",
    "get_run_id",
    "run_context",
    "set_correlation_id",
    "set_domain",
    "set_run_id",
]
