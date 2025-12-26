"""Correlation context propagation for observability."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass

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
_REPO: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_repo",
    default=None,
)
_COMMIT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_commit",
    default=None,
)


@dataclass(frozen=True, slots=True)
class CorrelationBundle:
    """Correlation identifiers for observability attributes."""

    correlation_id: str | None
    run_id: str | None
    domain: str | None
    repo: str | None
    commit: str | None

    def span_attributes(self) -> dict[str, str]:
        """Return span attributes derived from the bundle."""
        attrs: dict[str, str] = {}
        if self.correlation_id:
            attrs["codeintel.correlation_id"] = self.correlation_id
        if self.run_id:
            attrs["codeintel.run_id"] = self.run_id
        if self.domain:
            attrs["codeintel.domain"] = self.domain
        if self.repo:
            attrs["codeintel.repo"] = self.repo
        if self.commit:
            attrs["codeintel.commit"] = self.commit
        return attrs

    def metric_attributes(self) -> dict[str, str]:
        """Return low-cardinality metric attributes derived from the bundle."""
        attrs: dict[str, str] = {}
        if self.run_id:
            attrs["codeintel.run_id"] = self.run_id
        if self.domain:
            attrs["codeintel.domain"] = self.domain
        if self.repo:
            attrs["codeintel.repo"] = self.repo
        if self.commit:
            attrs["codeintel.commit"] = self.commit
        return attrs


def current_correlation_bundle() -> CorrelationBundle:
    """Return the current correlation bundle from context."""
    return CorrelationBundle(
        correlation_id=_CORRELATION_ID.get(),
        run_id=_RUN_ID.get(),
        domain=_DOMAIN.get(),
        repo=_REPO.get(),
        commit=_COMMIT.get(),
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
    """Set the current correlation identifier.

    Parameters
    ----------
    value
        Correlation identifier to set.
    """
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
    """Set the current run identifier.

    Parameters
    ----------
    value
        Run identifier to set.
    """
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
    """Set the current domain identifier.

    Parameters
    ----------
    value
        Domain identifier to set.
    """
    _DOMAIN.set(value)


def get_repo() -> str | None:
    """Return the current repository identifier.

    Returns
    -------
    str | None
        Repository identifier when set, otherwise ``None``.
    """
    return _REPO.get()


def set_repo(value: str | None) -> None:
    """Set the current repository identifier.

    Parameters
    ----------
    value
        Repository identifier to set.
    """
    _REPO.set(value)


def get_commit() -> str | None:
    """Return the current commit identifier.

    Returns
    -------
    str | None
        Commit identifier when set, otherwise ``None``.
    """
    return _COMMIT.get()


def set_commit(value: str | None) -> None:
    """Set the current commit identifier.

    Parameters
    ----------
    value
        Commit identifier to set.
    """
    _COMMIT.set(value)


@contextlib.contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Set the correlation identifier within a context manager.

    Parameters
    ----------
    correlation_id
        Correlation identifier to attach in context.
    """
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
def run_context(
    *,
    run_id: str | None = None,
    domain: str | None = None,
    repo: str | None = None,
    commit: str | None = None,
) -> Iterator[None]:
    """Set run and domain identifiers within a context manager.

    Parameters
    ----------
    run_id
        Run identifier to attach.
    domain
        Domain identifier to attach.
    repo
        Repository identifier to attach.
    commit
        Commit identifier to attach.
    """
    run_token = _RUN_ID.set(run_id)
    domain_token = _DOMAIN.set(domain)
    repo_token = _REPO.set(repo)
    commit_token = _COMMIT.set(commit)
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
    if repo:
        baggage = _otel_baggage.set_baggage(
            "codeintel.repo",
            repo,
            baggage,
        )
    if commit:
        baggage = _otel_baggage.set_baggage(
            "codeintel.commit",
            commit,
            baggage,
        )
    if baggage is not None:
        baggage_token = _otel_attach(baggage)

    try:
        yield
    finally:
        _RUN_ID.reset(run_token)
        _DOMAIN.reset(domain_token)
        _REPO.reset(repo_token)
        _COMMIT.reset(commit_token)
        if baggage_token is not None:
            _otel_detach(baggage_token)


__all__ = [
    "CorrelationBundle",
    "correlation_context",
    "get_commit",
    "get_correlation_id",
    "get_domain",
    "get_repo",
    "get_run_id",
    "current_correlation_bundle",
    "run_context",
    "set_commit",
    "set_correlation_id",
    "set_domain",
    "set_repo",
    "set_run_id",
]
