"""Tests for serving layer observability module.

This module tests the observability primitives for query services,
including metrics recording through the public API.
"""

from __future__ import annotations

import logging
from typing import cast

from codeintel.serving.context import RequestContext
from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    ServiceObservability,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_length,
    expect_not_in,
    expect_true,
)
from tests._helpers.fakes.logging import CAPTURE_HANDLER_LEVEL, CapturingHandler

# Constants for test values
DURATION_MS = 15.5
DURATION_PRECISE = 15.12345
DURATION_ROUNDED = 15.12
ROW_COUNT = 10
MESSAGE_COUNT_TWO = 2
ROW_COUNT_THREE = 3
ROW_COUNT_FIVE = 5


def _build_logger(
    name: str, *, level: int = logging.INFO
) -> tuple[logging.Logger, CapturingHandler]:
    """Construct a real logger with a capturing handler for assertions.

    Returns
    -------
    tuple[logging.Logger, CapturingHandler]
        Logger and attached handler collecting emitted records.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handler = CapturingHandler(level=CAPTURE_HANDLER_LEVEL)
    logger.handlers = [handler]
    return logger, handler


def _get_payload(handler: CapturingHandler, index: int = 0) -> dict[str, object]:
    """Extract the payload dict from a captured record.

    Returns
    -------
    dict[str, object]
        Structured payload emitted by ServiceObservability.
    """
    record = handler.records[index]
    args_obj = record.args
    if isinstance(args_obj, dict):
        payload: object = args_obj
    else:
        args = cast("tuple[object, ...]", args_obj)
        payload = args[0]
    return cast("dict[str, object]", payload)


# =============================================================================
# ServiceCallMetrics Tests
# =============================================================================


def test_service_call_metrics_required_fields() -> None:
    """Verify ServiceCallMetrics with only required fields."""
    metrics = ServiceCallMetrics(
        name="get_function_summary",
        transport="local",
        duration_ms=DURATION_MS,
    )

    expect_equal(metrics.name, "get_function_summary")
    expect_equal(metrics.transport, "local")
    expect_equal(metrics.duration_ms, DURATION_MS)


def test_service_call_metrics_all_fields() -> None:
    """Verify ServiceCallMetrics with all fields populated."""
    metrics = ServiceCallMetrics(
        name="list_datasets",
        transport="http",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT,
        dataset="analytics.functions",
        messages=MESSAGE_COUNT_TWO,
        error=None,
        truncated=False,
        schema_version="1.0.0",
        retries=1,
        correlation_id="corr-123",
        external_transport="mcp",
        operation="datasets.rows",
        repo="demo/repo",
        commit="deadbeef",
        client_id="client-abc",
        user_agent="CodeIntel/1.0",
    )

    expect_equal(metrics.rows, ROW_COUNT)
    expect_equal(metrics.dataset, "analytics.functions")
    expect_equal(metrics.messages, MESSAGE_COUNT_TWO)
    expect_false(metrics.truncated)
    expect_equal(metrics.schema_version, "1.0.0")
    expect_equal(metrics.retries, 1)
    expect_equal(metrics.correlation_id, "corr-123")


def test_service_call_metrics_with_error() -> None:
    """Verify ServiceCallMetrics records error information."""
    metrics = ServiceCallMetrics(
        name="get_function_summary",
        transport="local",
        duration_ms=DURATION_MS,
        error="ValueError",
    )

    expect_equal(metrics.error, "ValueError")


def test_service_call_metrics_optional_fields_none() -> None:
    """Verify ServiceCallMetrics optional fields default to None."""
    metrics = ServiceCallMetrics(
        name="test",
        transport="local",
        duration_ms=1.0,
    )

    expect_is_none(metrics.rows)
    expect_is_none(metrics.dataset)
    expect_is_none(metrics.messages)
    expect_is_none(metrics.error)
    expect_is_none(metrics.truncated)
    expect_is_none(metrics.schema_version)
    expect_is_none(metrics.retries)
    expect_is_none(metrics.correlation_id)
    expect_is_none(metrics.external_transport)
    expect_is_none(metrics.operation)
    expect_is_none(metrics.repo)
    expect_is_none(metrics.commit)
    expect_is_none(metrics.client_id)
    expect_is_none(metrics.user_agent)


# =============================================================================
# ServiceCallContext Tests
# =============================================================================


def test_service_call_context_defaults() -> None:
    """Verify ServiceCallContext default values."""
    ctx = ServiceCallContext()

    expect_is_none(ctx.dataset)
    expect_is_none(ctx.schema_version)
    expect_is_none(ctx.retries)


def test_service_call_context_with_values() -> None:
    """Verify ServiceCallContext with all values set."""
    ctx = ServiceCallContext(
        dataset="test.dataset",
        schema_version="2.0",
        retries=ROW_COUNT_THREE,
    )

    expect_equal(ctx.dataset, "test.dataset")
    expect_equal(ctx.schema_version, "2.0")
    expect_equal(ctx.retries, ROW_COUNT_THREE)


# =============================================================================
# ServiceObservability Tests
# =============================================================================


def test_service_observability_disabled_by_default() -> None:
    """Verify ServiceObservability is disabled by default."""
    obs = ServiceObservability()

    expect_false(obs.enabled)


def test_service_observability_enabled() -> None:
    """Verify ServiceObservability can be enabled."""
    obs = ServiceObservability(enabled=True)

    expect_true(obs.enabled)


def test_service_observability_custom_logger() -> None:
    """Verify ServiceObservability accepts custom logger."""
    custom_logger = logging.getLogger("custom.test")
    obs = ServiceObservability(enabled=True, logger=custom_logger)

    expect_true(obs.logger is custom_logger)


def test_service_observability_record_when_disabled() -> None:
    """Verify record does nothing when observability is disabled."""
    logger, handler = _build_logger("tests.observability.disabled")
    obs = ServiceObservability(enabled=False, logger=logger)

    metrics = ServiceCallMetrics(name="test", transport="local", duration_ms=1.0)
    obs.record(metrics)

    expect_length(handler.records, 0)


def test_service_observability_record_when_enabled() -> None:
    """Verify record logs when observability is enabled."""
    logger, handler = _build_logger("tests.observability.enabled")
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(
        name="test_call",
        transport="local",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT_FIVE,
    )
    obs.record(metrics)

    expect_length(handler.records, 1)
    record = handler.records[0]
    expect_true(record.getMessage().startswith("service_call"))
    payload = _get_payload(handler)
    expect_equal(payload["name"], "test_call")
    expect_equal(payload["rows"], ROW_COUNT_FIVE)


def test_service_observability_record_with_context() -> None:
    """Verify record includes RequestContext fields."""
    logger, handler = _build_logger("tests.observability.with_context")
    obs = ServiceObservability(enabled=True, logger=logger)

    ctx = RequestContext(
        correlation_id="ctx-123",
        transport="http",
        operation="datasets.rows",
        repo="test/repo",
        commit="abc123",
    )
    metrics = ServiceCallMetrics(
        name="test_call",
        transport="local",
        duration_ms=DURATION_MS,
    )
    obs.record(metrics, context=ctx)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)
    expect_equal(payload["correlation_id"], "ctx-123")
    expect_equal(payload["external_transport"], "http")
    expect_equal(payload["operation"], "datasets.rows")
    expect_equal(payload["repo"], "test/repo")


def test_service_observability_record_merges_context_values() -> None:
    """Verify record prefers metric values over context values."""
    logger, handler = _build_logger("tests.observability.merge_context")
    obs = ServiceObservability(enabled=True, logger=logger)

    ctx = RequestContext(
        correlation_id="ctx-fallback",
        transport="http",
    )
    metrics = ServiceCallMetrics(
        name="test",
        transport="local",
        duration_ms=1.0,
        correlation_id="metric-override",  # Should override ctx
    )
    obs.record(metrics, context=ctx)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)
    expect_equal(payload["correlation_id"], "metric-override")
    expect_equal(payload["external_transport"], "http")


def test_service_observability_record_all_optional_fields() -> None:
    """Verify record handles all optional metric fields."""
    logger, handler = _build_logger("tests.observability.optional_fields")
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(
        name="full_metrics",
        transport="http",
        duration_ms=DURATION_MS,
        rows=ROW_COUNT,
        dataset="test.dataset",
        messages=MESSAGE_COUNT_TWO,
        error=None,
        truncated=True,
        schema_version="1.0",
        retries=1,
    )
    obs.record(metrics)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)

    expect_equal(payload["rows"], ROW_COUNT)
    expect_equal(payload["dataset"], "test.dataset")
    expect_true(payload["truncated"])


def test_service_observability_record_with_error() -> None:
    """Verify record includes error information."""
    logger, handler = _build_logger("tests.observability.error")
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(
        name="error_call",
        transport="local",
        duration_ms=DURATION_MS,
        error="ValueError",
    )
    obs.record(metrics)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)
    expect_equal(payload["error"], "ValueError")


def test_service_observability_record_logger_not_enabled() -> None:
    """Verify record does nothing when logger level not enabled."""
    logger, handler = _build_logger("tests.observability.not_enabled", level=logging.ERROR)
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(name="test", transport="local", duration_ms=1.0)
    obs.record(metrics)

    expect_length(handler.records, 0)


def test_service_observability_record_context_enrichment() -> None:
    """Verify record enriches payload from RequestContext."""
    logger, handler = _build_logger("tests.observability.context_enrichment")
    obs = ServiceObservability(enabled=True, logger=logger)

    ctx = RequestContext(
        correlation_id="enrichment-test",
        transport="mcp",
        operation="get_function_summary",
        repo="demo/repo",
        commit="abc123",
        client_id="client-001",
        user_agent="TestAgent/1.0",
    )
    metrics = ServiceCallMetrics(
        name="context_test",
        transport="local",
        duration_ms=DURATION_MS,
    )
    obs.record(metrics, context=ctx)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)

    # Context values should be in payload
    expect_equal(payload["correlation_id"], "enrichment-test")
    expect_equal(payload["external_transport"], "mcp")
    expect_equal(payload["operation"], "get_function_summary")
    expect_equal(payload["repo"], "demo/repo")


def test_service_observability_record_rounds_duration() -> None:
    """Verify record rounds duration_ms to 2 decimal places."""
    logger, handler = _build_logger("tests.observability.rounding")
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(
        name="duration_test",
        transport="local",
        duration_ms=DURATION_PRECISE,
    )
    obs.record(metrics)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)

    # Duration should be rounded to 2 decimal places
    expect_equal(payload["duration_ms"], DURATION_ROUNDED)


def test_service_observability_record_excludes_none_values() -> None:
    """Verify record excludes optional fields that are None."""
    logger, handler = _build_logger("tests.observability.exclude_none")
    obs = ServiceObservability(enabled=True, logger=logger)

    metrics = ServiceCallMetrics(
        name="minimal",
        transport="local",
        duration_ms=1.0,
    )
    obs.record(metrics)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)

    # None values should not be in payload
    expect_not_in("rows", payload)
    expect_not_in("dataset", payload)
    expect_not_in("error", payload)
    expect_not_in("truncated", payload)


def test_service_observability_record_metric_overrides_context() -> None:
    """Verify metric values take precedence over context values."""
    logger, handler = _build_logger("tests.observability.metric_overrides")
    obs = ServiceObservability(enabled=True, logger=logger)

    ctx = RequestContext(
        correlation_id="context-corr",
        transport="http",
        repo="context-repo",
    )
    metrics = ServiceCallMetrics(
        name="override_test",
        transport="local",
        duration_ms=1.0,
        correlation_id="metric-corr",  # Should override context
        repo="metric-repo",  # Should override context
    )
    obs.record(metrics, context=ctx)

    expect_length(handler.records, 1)
    payload = _get_payload(handler)

    # Metric values should take precedence
    expect_equal(payload["correlation_id"], "metric-corr")
    expect_equal(payload["repo"], "metric-repo")


def test_service_observability_default_logger() -> None:
    """Verify ServiceObservability has default logger."""
    obs = ServiceObservability(enabled=True)

    expect_true(obs.logger is not None)
    expect_equal(obs.logger.name, "codeintel.serving.services.query")


def test_service_call_metrics_repo_commit_fields() -> None:
    """Verify ServiceCallMetrics supports repo and commit fields."""
    metrics = ServiceCallMetrics(
        name="repo_test",
        transport="http",
        duration_ms=1.0,
        repo="test/repository",
        commit="sha256hash",
    )

    expect_equal(metrics.repo, "test/repository")
    expect_equal(metrics.commit, "sha256hash")


def test_service_call_metrics_external_transport() -> None:
    """Verify ServiceCallMetrics external_transport field."""
    metrics = ServiceCallMetrics(
        name="transport_test",
        transport="local",
        duration_ms=1.0,
        external_transport="cli",
    )

    expect_equal(metrics.external_transport, "cli")
