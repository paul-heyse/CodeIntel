"""Tests for serving metrics logging utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def test_query_metrics_dataclass_is_frozen() -> None:
    """QueryMetrics should be immutable."""
    metrics = QueryMetrics(
        endpoint="/v1/semantic/query",
        view_id="demo.view",
        query=None,
        row_count=10,
        truncated=False,
        duration_ms=5.5,
        correlation_id="cid-123",
        engine="polars",
    )
    expect_equal(metrics.endpoint, "/v1/semantic/query")
    expect_equal(metrics.view_id, "demo.view")
    expect_equal(metrics.row_count, 10)
    expect_false(metrics.truncated)
    expect_equal(metrics.duration_ms, 5.5)
    expect_equal(metrics.correlation_id, "cid-123")
    expect_equal(metrics.engine, "polars")


def test_query_metrics_default_engine() -> None:
    """Engine field should default to None."""
    metrics = QueryMetrics(
        endpoint="/v1/search",
        view_id=None,
        query="test query",
        row_count=5,
        truncated=False,
        duration_ms=2.0,
        correlation_id="cid-456",
    )
    expect_equal(metrics.engine, None)


def test_log_query_metrics_logs_structured_data(caplog: LogCaptureFixture) -> None:
    """log_query_metrics should emit structured log with expected fields."""
    metrics = QueryMetrics(
        endpoint="/v1/semantic/query",
        view_id="function.summary",
        query=None,
        row_count=100,
        truncated=True,
        duration_ms=15.123456,
        correlation_id="cid-789",
        engine="polars",
    )

    with caplog.at_level(logging.INFO, logger="codeintel.serving.metrics"):
        log_query_metrics(metrics)

    expect_equal(len(caplog.records), 1)
    record = caplog.records[0]

    expect_equal(record.levelname, "INFO")
    expect_in("query_executed", record.message)

    expect_equal(getattr(record, "endpoint", None), "/v1/semantic/query")
    expect_equal(getattr(record, "view_id", None), "function.summary")
    expect_equal(getattr(record, "query", None), None)
    expect_equal(getattr(record, "row_count", None), 100)
    expect_true(getattr(record, "truncated", False))
    expect_equal(getattr(record, "duration_ms", None), 15.123)
    expect_equal(getattr(record, "correlation_id", None), "cid-789")
    expect_equal(getattr(record, "engine", None), "polars")


def test_log_query_metrics_rounds_duration(caplog: LogCaptureFixture) -> None:
    """Duration should be rounded to 3 decimal places."""
    metrics = QueryMetrics(
        endpoint="/v1/search",
        view_id=None,
        query="test",
        row_count=1,
        truncated=False,
        duration_ms=1.23456789,
        correlation_id="cid-test",
    )

    with caplog.at_level(logging.INFO, logger="codeintel.serving.metrics"):
        log_query_metrics(metrics)

    record = caplog.records[0]
    expect_equal(getattr(record, "duration_ms", None), 1.235)


def test_log_query_metrics_search_endpoint(caplog: LogCaptureFixture) -> None:
    """Search endpoints should include query text."""
    metrics = QueryMetrics(
        endpoint="/v1/search",
        view_id=None,
        query="authentication handler",
        row_count=25,
        truncated=False,
        duration_ms=8.5,
        correlation_id="cid-search",
        engine="pandas",
    )

    with caplog.at_level(logging.INFO, logger="codeintel.serving.metrics"):
        log_query_metrics(metrics)

    record = caplog.records[0]
    expect_equal(getattr(record, "query", None), "authentication handler")
    expect_equal(getattr(record, "engine", None), "pandas")
    expect_true(getattr(record, "view_id", None) is None)
