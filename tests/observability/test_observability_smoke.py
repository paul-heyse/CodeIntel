"""Observability smoke tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi import BackgroundTasks
from fastmcp.server.middleware.middleware import MiddlewareContext
from starlette.requests import Request

from codeintel.cli.config.model import CliConfig, TelemetryConfig
from codeintel.cli.execution.bootstrap import bootstrap_cli, reset_bootstrap
from codeintel.observability.operation_scope import observe_operation, record_query_metrics
from codeintel.observability.runtime import (
    MetricConfig,
    ObservabilityConfig,
    ResourceConfig,
    TraceConfig,
    bootstrap_observability,
    shutdown_observability,
)
from codeintel.observability.telemetry_context import (
    RepoCommitContext,
    telemetry_context,
)
from codeintel.serving.http.route_utils import (
    ThreadpoolMetricsContext,
    run_in_threadpool_with_metrics,
)
from codeintel.serving.mcp.middleware_stack import McpOpenTelemetryMiddleware
from codeintel.serving.metrics import QueryMetrics
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)
from tests._helpers.env import temporary_env

trace_api = pytest.importorskip("opentelemetry.trace")
sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_CORRELATION_ID_HEX_LENGTH = 32


@dataclass(frozen=True)
class _DummyQueryMetrics:
    endpoint: str
    duration_ms: float
    row_count: int
    truncated: bool
    view_id: str | None
    query_hash: str | None
    schema_hash: str | None


def _configure_tracing() -> InMemorySpanExporter:
    bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            resources=ResourceConfig(service_name="codeintel-test"),
            traces=TraceConfig(enabled=False, console_export=False),
            metrics=MetricConfig(enabled=False, prometheus_enabled=False),
        )
    )
    exporter = in_memory.InMemorySpanExporter()
    provider = trace_api.get_tracer_provider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    return exporter


def _span_attributes(span: object) -> Mapping[str, object]:
    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def test_observe_operation_emits_span() -> None:
    """Ensure observe_operation emits a span."""
    shutdown_observability()
    exporter = _configure_tracing()

    with observe_operation(component="cli", operation="health"):
        pass

    spans = exporter.get_finished_spans()
    expect_true(bool(spans), message="Expected spans to be recorded")
    expect_equal(spans[-1].name, "cli.health")


def test_record_query_metrics_smoke() -> None:
    """Ensure query metrics recording does not error."""
    shutdown_observability()
    _ = _configure_tracing()

    record_query_metrics(
        _DummyQueryMetrics(
            endpoint="/v1/semantic/query",
            duration_ms=12.5,
            row_count=2,
            truncated=False,
            view_id=None,
            query_hash=None,
            schema_hash=None,
        )
    )


def test_observe_operation_includes_repo_and_commit() -> None:
    """Ensure run context repo/commit attributes land on spans."""
    shutdown_observability()
    exporter = _configure_tracing()

    with (
        telemetry_context(
            run_id="run-1",
            domain="tests",
            repo_commit=RepoCommitContext(
                repo="org/repo",
                commit="abc123",
            ),
        ),
        observe_operation(
            component="cli",
            operation="health",
        ),
    ):
        pass

    spans = exporter.get_finished_spans()
    expect_true(bool(spans), message="Expected spans to be recorded")
    attrs = _span_attributes(spans[-1])
    expect_equal(attrs.get("codeintel.repo"), "org/repo")
    expect_equal(attrs.get("codeintel.commit"), "abc123")


def test_cli_bootstrap_emits_span() -> None:
    """Ensure CLI bootstrap initializes tracing."""
    with temporary_env(CODEINTEL_TEST_TELEMETRY_MODE="inherit"):
        shutdown_observability()
        reset_bootstrap()

        config = CliConfig(
            log_level="WARNING",
            telemetry=TelemetryConfig(
                enabled=True,
                service_name="codeintel-cli-test",
                endpoint=None,
            ),
        )

        bootstrap_cli(config=config)

        exporter = in_memory.InMemorySpanExporter()
        provider = trace_api.get_tracer_provider()
        provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))

        with observe_operation(component="cli", operation="bootstrap"):
            pass

        spans = exporter.get_finished_spans()
        expect_true(bool(spans), message="Expected spans to be recorded")
        expect_equal(spans[-1].name, "cli.bootstrap")
        reset_bootstrap()


@pytest.mark.asyncio
async def test_http_route_wrapper_emits_span() -> None:
    """Ensure HTTP route wrapper emits a span."""
    shutdown_observability()
    exporter = _configure_tracing()

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/semantic/query",
        "query_string": b"",
        "headers": [],
    }
    request = Request(scope)
    request.state.correlation_id = "cid-123"
    request.scope["route"] = SimpleNamespace(path="/v1/semantic/query")

    background = BackgroundTasks()

    def _fn() -> str:
        return "ok"

    def _success_metrics(
        _result: str,
        duration_ms: float,
        correlation_id: str,
    ) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/query",
            view_id=None,
            query=None,
            row_count=1,
            truncated=False,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )

    def _error_metrics(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/query",
            view_id=None,
            query=None,
            row_count=0,
            truncated=False,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )

    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success_metrics,
        error_metrics=_error_metrics,
    )
    result = await run_in_threadpool_with_metrics(context, _fn)
    expect_equal(result, "ok")

    spans = exporter.get_finished_spans()
    expect_true(bool(spans), message="Expected spans to be recorded")
    expect_equal(spans[-1].name, "http./v1/semantic/query")


@pytest.mark.asyncio
async def test_mcp_middleware_emits_span() -> None:
    """Ensure MCP middleware emits a span."""
    shutdown_observability()
    exporter = _configure_tracing()

    middleware = McpOpenTelemetryMiddleware()
    context: MiddlewareContext[object] = MiddlewareContext(
        message=SimpleNamespace(name="semantic_query"),
        fastmcp_context=None,
        method="tools/call",
    )

    async def _call_next(context: MiddlewareContext[object]) -> str:
        _ = context
        await asyncio.sleep(0)
        return "ok"

    result = await middleware.on_message(context, _call_next)
    expect_equal(result, "ok")

    spans = exporter.get_finished_spans()
    expect_true(bool(spans), message="Expected spans to be recorded")
    span = spans[-1]
    expect_equal(span.name, "mcp.tools/call:semantic_query")
    attrs = _span_attributes(span)
    correlation_id = attrs.get("codeintel.correlation_id")
    if not isinstance(correlation_id, str):
        pytest.fail("Expected string correlation id")
    correlation_value = correlation_id
    expect_true(
        len(correlation_value) == _CORRELATION_ID_HEX_LENGTH,
        message="Expected hex correlation id",
    )
