"""RequestContext propagation into service observability metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.serving.context import (
    RequestContext,
    reset_current_request_context,
    set_current_request_context,
)
from codeintel.serving.services.observability import (
    ServiceCallMetrics,
    ServiceObservability,
)
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service


class CapturingObservability(ServiceObservability):
    """Observability sink that records metrics and the passed context."""

    def __init__(self) -> None:
        logger = logging.getLogger("test_request_context_observability")
        logger.setLevel(logging.INFO)
        super().__init__(enabled=True, logger=logger)
        self.events: list[tuple[ServiceCallMetrics, RequestContext | None]] = []

    def record(self, metrics: ServiceCallMetrics, context: RequestContext | None = None) -> None:
        """Capture metrics and context for assertions."""
        self.events.append((metrics, context))


def _build_local_service(
    gateway: StorageGateway,
    *,
    observability: ServiceObservability,
) -> LocalQueryService:
    limits = BackendLimits(default_limit=3, max_rows_per_call=5)
    query = build_duckdb_query_service(gateway, repo="demo/repo", commit="deadbeef", limits=limits)
    return LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
        observability=observability,
    )


def _with_request_context(ctx: RequestContext, func: Callable[[], object]) -> None:
    token = set_current_request_context(ctx)
    try:
        func()
    finally:
        reset_current_request_context(token)


def test_local_query_observability_uses_request_context(
    architecture_gateway: StorageGateway,
) -> None:
    """LocalQueryService should project RequestContext fields into metrics."""
    obs = CapturingObservability()
    service = _build_local_service(architecture_gateway, observability=obs)

    ctx = RequestContext(
        correlation_id="test-cid-123",
        transport="http",
        operation=None,
        dataset=None,
        repo="demo/repo",
        commit="deadbeef",
        snapshot=None,
        graph_scope=None,
        client_id="test-client",
        user_agent="pytest",
    )

    _with_request_context(ctx, lambda: service.list_subsystems(limit=1))

    if len(obs.events) != 1:
        message = f"Expected a single observability event, saw {len(obs.events)}"
        pytest.fail(message)
    metrics, recorded_ctx = obs.events[0]
    if metrics.correlation_id != ctx.correlation_id:
        pytest.fail(f"Correlation id missing from metrics: {metrics}")
    if metrics.external_transport != ctx.transport:
        pytest.fail(f"Transport missing from metrics: {metrics}")
    if metrics.operation != metrics.name:
        pytest.fail(f"Operation fallback incorrect: {metrics}")
    if metrics.repo != ctx.repo:
        pytest.fail(f"Repo missing from metrics: {metrics}")
    if metrics.commit != ctx.commit:
        pytest.fail(f"Commit missing from metrics: {metrics}")
    if metrics.client_id != ctx.client_id:
        pytest.fail(f"Client id missing from metrics: {metrics}")
    if metrics.user_agent != ctx.user_agent:
        pytest.fail(f"User agent missing from metrics: {metrics}")
    if recorded_ctx is not ctx:
        pytest.fail("Observability sink did not receive the active RequestContext")
