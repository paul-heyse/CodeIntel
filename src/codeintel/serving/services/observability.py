"""Observability primitives for serving query services."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from codeintel.serving import domain_models as dm
from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    FileHintsResponse,
    HighRiskFunctionsResponse,
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.services.errors import ProblemError

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.context import RequestContext

LOG = logging.getLogger("codeintel.serving.services.query")


@dataclass
class ServiceCallMetrics:
    """Structured metrics describing a service invocation."""

    name: str
    transport: str
    duration_ms: float
    rows: int | None = None
    dataset: str | None = None
    messages: int | None = None
    error: str | None = None
    truncated: bool | None = None
    schema_version: str | None = None
    retries: int | None = None
    correlation_id: str | None = None
    external_transport: str | None = None
    operation: str | None = None
    repo: str | None = None
    commit: str | None = None
    client_id: str | None = None
    user_agent: str | None = None


@dataclass
class ServiceCallContext:
    """Context propagated into observability signals."""

    dataset: str | None = None
    schema_version: str | None = None
    retries: int | None = None


@dataclass
class ServiceObservability:
    """Configuration for service-level observability."""

    enabled: bool = False
    logger: logging.Logger = field(default_factory=lambda: LOG)

    def record(
        self,
        metrics: ServiceCallMetrics,
        context: RequestContext | None = None,
    ) -> None:
        """
        Emit a structured log line for a service call.

        Parameters
        ----------
        metrics
            Call metrics describing the invocation outcome.
        context
            Optional RequestContext to enrich the payload.
        """
        if not self.enabled or not self.logger.isEnabledFor(logging.INFO):
            return
        payload: dict[str, object] = {
            "name": metrics.name,
            "transport": metrics.transport,
            "duration_ms": round(metrics.duration_ms, 2),
        }

        def _add_optional(key: str, value: object | None) -> None:
            if value is not None:
                payload[key] = value

        for key, value in (
            ("rows", metrics.rows),
            ("dataset", metrics.dataset),
            ("messages", metrics.messages),
            ("error", metrics.error),
            ("truncated", metrics.truncated),
            ("schema_version", metrics.schema_version),
            ("retries", metrics.retries),
        ):
            _add_optional(key, value)

        ctx = context

        def _context_value(metric_value: object | None, fallback: object | None) -> object | None:
            return metric_value if metric_value is not None else fallback

        context_pairs = (
            (
                "correlation_id",
                _context_value(metrics.correlation_id, ctx.correlation_id if ctx else None),
            ),
            (
                "external_transport",
                _context_value(metrics.external_transport, ctx.transport if ctx else None),
            ),
            (
                "operation",
                _context_value(metrics.operation, ctx.operation if ctx else None),
            ),
            ("repo", _context_value(metrics.repo, ctx.repo if ctx else None)),
            ("commit", _context_value(metrics.commit, ctx.commit if ctx else None)),
            ("client_id", _context_value(metrics.client_id, ctx.client_id if ctx else None)),
            ("user_agent", _context_value(metrics.user_agent, ctx.user_agent if ctx else None)),
        )
        for key, value in context_pairs:
            _add_optional(key, value)
        self.logger.info("service_call %s", payload)


def _infer_row_count(result: object) -> int | None:
    """
    Attempt to derive a row count from common response shapes.

    Returns
    -------
    int | None
        Row count when inferrable; otherwise ``None``.
    """
    attr_counts: list[tuple[type, str]] = [
        (DatasetRowsResponse, "rows"),
        (dm.DatasetRows, "rows"),
        (HighRiskFunctionsResponse, "functions"),
        (dm.HighRiskFunctionsResult, "functions"),
        (TestsForFunctionResponse, "tests"),
        (dm.TestsForFunctionResult, "tests"),
        (SubsystemSummaryResponse, "subsystems"),
        (dm.SubsystemSummaryResult, "subsystems"),
        (ModuleSubsystemResponse, "memberships"),
        (dm.ModuleSubsystemResult, "memberships"),
        (FileHintsResponse, "hints"),
        (dm.FileHintsResult, "hints"),
        (SubsystemModulesResponse, "modules"),
        (dm.SubsystemModulesResult, "modules"),
        (SubsystemSearchResponse, "subsystems"),
        (dm.SubsystemSearchResult, "subsystems"),
        (SubsystemProfileResponse, "profiles"),
        (dm.SubsystemProfileResult, "profiles"),
        (SubsystemCoverageResponse, "coverage"),
        (dm.SubsystemCoverageResult, "coverage"),
    ]
    if isinstance(result, CallGraphNeighborsResponse):
        return len(result.outgoing) + len(result.incoming)
    for response_type, attr in attr_counts:
        if isinstance(result, response_type):
            typed_result = cast("Any", result)
            return len(getattr(typed_result, attr))
    return None


def _extract_message_count(result: object) -> int | None:
    """
    Return the number of response messages when available.

    Returns
    -------
    int | None
        Message count if present; otherwise ``None``.
    """
    meta = getattr(result, "meta", None)
    if meta is None or meta.messages is None:
        return None
    return len(meta.messages)


def _extract_truncated(result: object) -> bool | None:
    """
    Return truncation state when available on response metadata.

    Returns
    -------
    bool | None
        Truncation flag if present; otherwise ``None``.
    """
    meta = getattr(result, "meta", None)
    if meta is None:
        return None
    truncated = getattr(meta, "truncated", None)
    return bool(truncated) if truncated is not None else None


def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    context: ServiceCallContext | None,
    func: Callable[[], T],
) -> T:
    """
    Execute a callable while capturing observability signals.

    Uses the current RequestContext (if any) to enrich metrics.

    Returns
    -------
    T
        Result returned by the wrapped callable.

    Raises
    ------
    ProblemError
        When the wrapped callable surfaces a domain problem.
    RuntimeError
        When the callable signals runtime failures.
    ValueError
        When the callable surfaces invalid inputs.
    OSError
        When I/O issues occur within the callable.
    TimeoutError
        When the callable indicates it exceeded a timeout.
    """
    req_ctx = get_current_request_context()
    start = time.perf_counter()

    def _apply_request_context(metrics: ServiceCallMetrics) -> None:
        if req_ctx is None:
            return
        metrics.correlation_id = req_ctx.correlation_id
        metrics.external_transport = req_ctx.transport
        metrics.operation = req_ctx.operation or name
        metrics.repo = req_ctx.repo
        metrics.commit = req_ctx.commit
        metrics.client_id = req_ctx.client_id
        metrics.user_agent = req_ctx.user_agent

    try:
        result = func()
    except (ProblemError, RuntimeError, ValueError, OSError, TimeoutError) as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        if observability is not None:
            metrics = ServiceCallMetrics(
                name=name,
                transport=transport,
                duration_ms=duration_ms,
                dataset=context.dataset if context is not None else None,
                schema_version=context.schema_version if context is not None else None,
                retries=context.retries if context is not None else None,
                error=exc.__class__.__name__,
            )
            _apply_request_context(metrics)
            observability.record(metrics, context=req_ctx)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    if observability is not None:
        metrics = ServiceCallMetrics(
            name=name,
            transport=transport,
            duration_ms=duration_ms,
            rows=_infer_row_count(result),
            dataset=context.dataset if context is not None else None,
            messages=_extract_message_count(result),
            truncated=_extract_truncated(result),
            schema_version=context.schema_version if context is not None else None,
            retries=context.retries if context is not None else None,
        )
        _apply_request_context(metrics)
        observability.record(metrics, context=req_ctx)
    return result


__all__ = [
    "LOG",
    "ServiceCallContext",
    "ServiceCallMetrics",
    "ServiceObservability",
    "_observe_call",
]
