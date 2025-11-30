"""Observability primitives for serving query services."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from codeintel.serving import domain_models as dm
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

    def record(self, metrics: ServiceCallMetrics) -> None:
        """
        Emit a structured log line for a service call.

        Parameters
        ----------
        metrics
            Call metrics describing the invocation outcome.
        """
        if not self.enabled or not self.logger.isEnabledFor(logging.INFO):
            return
        payload: dict[str, object] = {
            "name": metrics.name,
            "transport": metrics.transport,
            "duration_ms": round(metrics.duration_ms, 2),
        }
        if metrics.rows is not None:
            payload["rows"] = metrics.rows
        if metrics.dataset is not None:
            payload["dataset"] = metrics.dataset
        if metrics.messages is not None:
            payload["messages"] = metrics.messages
        if metrics.error is not None:
            payload["error"] = metrics.error
        if metrics.truncated is not None:
            payload["truncated"] = metrics.truncated
        if metrics.schema_version is not None:
            payload["schema_version"] = metrics.schema_version
        if metrics.retries is not None:
            payload["retries"] = metrics.retries
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

    Returns
    -------
    T
        Result returned by the wrapped callable.
    """
    start = time.perf_counter()
    try:
        result = func()
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        if observability is not None:
            observability.record(
                ServiceCallMetrics(
                    name=name,
                    transport=transport,
                    duration_ms=duration_ms,
                    dataset=context.dataset if context is not None else None,
                    schema_version=context.schema_version if context is not None else None,
                    retries=context.retries if context is not None else None,
                    error=exc.__class__.__name__,
                )
            )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    if observability is not None:
        observability.record(
            ServiceCallMetrics(
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
        )
    return result


__all__ = [
    "LOG",
    "ServiceCallContext",
    "ServiceCallMetrics",
    "ServiceObservability",
    "_observe_call",
]
