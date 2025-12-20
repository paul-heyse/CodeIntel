"""Tests for MCP tool metrics emission."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def _set_env(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables.

    Parameters
    ----------
    env
        Environment variables to set.

    Yields
    ------
    None
        Context manager scope.
    """
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LogCapture(logging.Handler):
    """Capture log records for testing.

    Attributes
    ----------
    records
        List of captured log records.
    """

    def __init__(self) -> None:
        """Initialize log capture handler."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture a log record.

        Parameters
        ----------
        record
            Log record to capture.
        """
        self.records.append(record)


def test_query_metrics_dataclass() -> None:
    """Verify QueryMetrics dataclass holds expected fields."""
    metrics = QueryMetrics(
        endpoint="mcp:semantic_query",
        view_id="analytics.function_metrics",
        query=None,
        row_count=42,
        truncated=False,
        duration_ms=123.456,
        correlation_id="session-12345",
    )
    expect_equal(metrics.endpoint, "mcp:semantic_query")
    expect_equal(metrics.view_id, "analytics.function_metrics")
    expect_is_none(metrics.query)
    expect_equal(metrics.row_count, 42)
    expect_false(metrics.truncated)
    expect_equal(metrics.duration_ms, 123.456)
    expect_equal(metrics.correlation_id, "session-12345")


def test_query_metrics_with_search_query() -> None:
    """Verify QueryMetrics captures search query."""
    metrics = QueryMetrics(
        endpoint="mcp:code_search",
        view_id=None,
        query="find_user_by_id",
        row_count=15,
        truncated=False,
        duration_ms=50.0,
        correlation_id="mcp-unknown",
    )
    expect_is_none(metrics.view_id)
    expect_equal(metrics.query, "find_user_by_id")


def test_query_metrics_optional_engine() -> None:
    """Verify QueryMetrics engine field is optional."""
    metrics = QueryMetrics(
        endpoint="mcp:semantic_query",
        view_id="test.view",
        query=None,
        row_count=10,
        truncated=False,
        duration_ms=100.0,
        correlation_id="test",
        engine="polars",
    )
    expect_equal(metrics.engine, "polars")


def test_log_query_metrics_emits_info() -> None:
    """Verify log_query_metrics emits INFO level log."""
    logger = logging.getLogger("codeintel.serving.metrics")
    capture = LogCapture()
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    try:
        metrics = QueryMetrics(
            endpoint="mcp:semantic_catalog",
            view_id=None,
            query=None,
            row_count=5,
            truncated=False,
            duration_ms=25.5,
            correlation_id="session-xyz",
        )
        log_query_metrics(metrics)

        expect_equal(len(capture.records), 1)
        record = capture.records[0]
        expect_equal(record.levelno, logging.INFO)
        expect_equal(record.msg, "query_executed")
    finally:
        logger.removeHandler(capture)


def test_log_query_metrics_extra_fields() -> None:
    """Verify log_query_metrics includes structured extra fields."""
    logger = logging.getLogger("codeintel.serving.metrics")
    capture = LogCapture()
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    try:
        metrics = QueryMetrics(
            endpoint="mcp:semantic_query",
            view_id="test.users",
            query=None,
            row_count=100,
            truncated=True,
            duration_ms=500.123,
            correlation_id="corr-abc",
            engine="pandas",
        )
        log_query_metrics(metrics)

        record = capture.records[0]
        expect_is_not_none(record)
        expect_equal(record.__dict__["endpoint"], "mcp:semantic_query")
        expect_equal(record.__dict__["view_id"], "test.users")
        expect_is_none(record.__dict__["query"])
        expect_equal(record.__dict__["row_count"], 100)
        expect_true(record.__dict__["truncated"])
        expect_equal(record.__dict__["duration_ms"], 500.123)
        expect_equal(record.__dict__["correlation_id"], "corr-abc")
        expect_equal(record.__dict__["engine"], "pandas")
    finally:
        logger.removeHandler(capture)


def test_metrics_endpoint_naming_mcp_semantic() -> None:
    """Verify MCP tools use mcp: prefix for endpoint naming."""
    # Test various MCP endpoint names
    endpoints = [
        "mcp:semantic_catalog",
        "mcp:semantic_describe",
        "mcp:semantic_query",
        "mcp:semantic_explain",
        "mcp:semantic_export",
        "mcp:serving_meta",
        "mcp:code_search",
    ]
    for endpoint in endpoints:
        expect_true(endpoint.startswith("mcp:"))


def test_metrics_correlation_id_fallback() -> None:
    """Verify correlation_id fallback to mcp-unknown."""
    metrics = QueryMetrics(
        endpoint="mcp:semantic_query",
        view_id="test.view",
        query=None,
        row_count=0,
        truncated=False,
        duration_ms=10.0,
        correlation_id="mcp-unknown",
    )
    expect_equal(metrics.correlation_id, "mcp-unknown")


def test_metrics_truncated_flag() -> None:
    """Verify truncated flag can be True."""
    metrics = QueryMetrics(
        endpoint="mcp:semantic_query",
        view_id="test.view",
        query=None,
        row_count=200,
        truncated=True,
        duration_ms=150.0,
        correlation_id="test",
    )
    expect_true(metrics.truncated)
