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

from codeintel.observability.semconv_keys import (
    CODEINTEL_ACTOR,
    CODEINTEL_COMMIT,
    CODEINTEL_CORRELATION_ID,
    CODEINTEL_DOMAIN,
    CODEINTEL_REPO,
    CODEINTEL_RUN_ID,
)

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
_ACTOR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "codeintel_actor",
    default=None,
)


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    """Correlation identifiers for observability attributes."""

    correlation_id: str | None
    run_id: str | None
    domain: str | None
    repo: str | None
    commit: str | None
    actor: str | None

    def span_attributes(self) -> dict[str, str]:
        """Return span attributes derived from the context.

        Returns
        -------
        dict[str, str]
            Span attributes for the current context.
        """
        attrs: dict[str, str] = {}
        if self.correlation_id:
            attrs[CODEINTEL_CORRELATION_ID] = self.correlation_id
        if self.run_id:
            attrs[CODEINTEL_RUN_ID] = self.run_id
        if self.domain:
            attrs[CODEINTEL_DOMAIN] = self.domain
        if self.repo:
            attrs[CODEINTEL_REPO] = self.repo
        if self.commit:
            attrs[CODEINTEL_COMMIT] = self.commit
        if self.actor:
            attrs[CODEINTEL_ACTOR] = self.actor
        return attrs

    def metric_attributes(self) -> dict[str, str]:
        """Return low-cardinality metric attributes derived from the context.

        Returns
        -------
        dict[str, str]
            Metric attributes for the current context.
        """
        attrs: dict[str, str] = {}
        if self.run_id:
            attrs[CODEINTEL_RUN_ID] = self.run_id
        if self.domain:
            attrs[CODEINTEL_DOMAIN] = self.domain
        if self.repo:
            attrs[CODEINTEL_REPO] = self.repo
        if self.commit:
            attrs[CODEINTEL_COMMIT] = self.commit
        return attrs


@dataclass(frozen=True, slots=True)
class RepoCommitContext:
    """Repository identity context for telemetry attributes."""

    repo: str | None
    commit: str | None


def current_telemetry_context() -> TelemetryContext:
    """Return the current telemetry context from context variables.

    Returns
    -------
    TelemetryContext
        Current telemetry correlation identifiers.
    """
    return TelemetryContext(
        correlation_id=_CORRELATION_ID.get(),
        run_id=_RUN_ID.get(),
        domain=_DOMAIN.get(),
        repo=_REPO.get(),
        commit=_COMMIT.get(),
        actor=_ACTOR.get(),
    )


@contextlib.contextmanager
def telemetry_context(
    *,
    correlation_id: str | None = None,
    run_id: str | None = None,
    domain: str | None = None,
    repo_commit: RepoCommitContext | None = None,
    actor: str | None = None,
) -> Iterator[None]:
    """Attach telemetry correlation identifiers within a context manager."""
    repo = repo_commit.repo if repo_commit is not None else None
    commit = repo_commit.commit if repo_commit is not None else None
    correlation_token = _CORRELATION_ID.set(correlation_id)
    run_token = _RUN_ID.set(run_id)
    domain_token = _DOMAIN.set(domain)
    repo_token = _REPO.set(repo)
    commit_token = _COMMIT.set(commit)
    actor_token = _ACTOR.set(actor)
    baggage_token: contextvars.Token[Context] | None = None

    baggage = None
    if correlation_id:
        baggage = _otel_baggage.set_baggage(CODEINTEL_CORRELATION_ID, correlation_id)
    if run_id:
        baggage = _otel_baggage.set_baggage(
            CODEINTEL_RUN_ID,
            run_id,
            baggage,
        )
    if domain:
        baggage = _otel_baggage.set_baggage(
            CODEINTEL_DOMAIN,
            domain,
            baggage,
        )
    if repo:
        baggage = _otel_baggage.set_baggage(
            CODEINTEL_REPO,
            repo,
            baggage,
        )
    if commit:
        baggage = _otel_baggage.set_baggage(
            CODEINTEL_COMMIT,
            commit,
            baggage,
        )
    if actor:
        baggage = _otel_baggage.set_baggage(
            CODEINTEL_ACTOR,
            actor,
            baggage,
        )
    if baggage is not None:
        baggage_token = _otel_attach(baggage)

    try:
        yield
    finally:
        _CORRELATION_ID.reset(correlation_token)
        _RUN_ID.reset(run_token)
        _DOMAIN.reset(domain_token)
        _REPO.reset(repo_token)
        _COMMIT.reset(commit_token)
        _ACTOR.reset(actor_token)
        if baggage_token is not None:
            _otel_detach(baggage_token)


__all__ = [
    "RepoCommitContext",
    "TelemetryContext",
    "current_telemetry_context",
    "telemetry_context",
]
