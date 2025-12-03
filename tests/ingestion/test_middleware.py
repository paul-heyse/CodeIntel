"""Tests for ingestion middleware chain and implementations.

This module tests the middleware protocol, chain execution, and all
concrete middleware implementations (logging, metrics, tracing).

Enhanced with realistic runtime conditions using:
- Real plugins from DEFAULT_INGEST_PLUGINS (RepoScanPlugin, AstExtractPlugin)
- IngestTestSetup.from_repo() for production-parity context building
- provisioned_repo fixture for seeded gateway state
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.core.middleware import (
    IngestMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    TracingMiddleware,
)
from codeintel.ingestion.core.middleware.metrics import InMemoryMetrics
from codeintel.ingestion.core.middleware.tracing import InMemoryTracer
from codeintel.ingestion.plugins import (
    AstExtractPlugin,
    IngestPluginResult,
    RepoScanPlugin,
)
from tests._helpers.fixtures import ProvisionedGateway
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway
from tests._helpers.ingest_setup import IngestTestSetup

# Test constants
EXPECTED_ROW_COUNT_SUM = 50
EXPECTED_ROW_COUNT_SINGLE = 42
MIDDLEWARE_FAILURE_MSG = "Middleware failure"


# =============================================================================
# Protocol-Based Test Doubles (per Testing Charter - legitimate pattern)
# =============================================================================


@dataclass
class RecordingMiddleware:
    """Middleware that records all calls for testing.

    This is a legitimate protocol-based test double that implements
    IngestMiddleware to record execution flow without modifying behavior.
    """

    calls: list[tuple[str, str]] = field(default_factory=list)
    should_fail: bool = False

    def before_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Record before_execute call.

        Parameters
        ----------
        plugin
            Plugin being executed.
        ctx
            Execution context (unused in recording).

        Raises
        ------
        ValueError
            If should_fail is True.
        """
        _ = ctx
        self.calls.append(("before", plugin.metadata.name))
        if self.should_fail:
            raise ValueError(MIDDLEWARE_FAILURE_MSG)

    def after_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """Record after_execute call.

        Parameters
        ----------
        plugin
            Plugin that executed.
        ctx
            Execution context (unused in recording).
        result
            Execution result (unused in recording).

        Raises
        ------
        ValueError
            If should_fail is True.
        """
        _ = ctx
        _ = result
        self.calls.append(("after", plugin.metadata.name))
        if self.should_fail:
            raise ValueError(MIDDLEWARE_FAILURE_MSG)

    def on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """Record on_error call.

        Parameters
        ----------
        plugin
            Plugin that failed.
        ctx
            Execution context (unused in recording).
        error
            Exception that was raised (unused in recording).

        Raises
        ------
        ValueError
            If should_fail is True.
        """
        _ = ctx
        _ = error
        self.calls.append(("error", plugin.metadata.name))
        if self.should_fail:
            raise ValueError(MIDDLEWARE_FAILURE_MSG)


# =============================================================================
# Fixture Helpers Using Production Infrastructure
# =============================================================================


def _create_test_repo(repo_root: Path) -> None:
    """Create a minimal Python package for realistic plugin testing.

    Parameters
    ----------
    repo_root
        Root directory for the test repository.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        '"""Test package for middleware integration tests."""\n',
        encoding="utf-8",
    )

    (pkg_dir / "mod.py").write_text(
        '''"""Sample module for testing."""


def greet(name: str) -> str:
    """Return a greeting message.

    Parameters
    ----------
    name
        Name to greet.

    Returns
    -------
    str
        Greeting message.
    """
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two integers.

    Parameters
    ----------
    a
        First operand.
    b
        Second operand.

    Returns
    -------
    int
        Sum of a and b.
    """
    return a + b
''',
        encoding="utf-8",
    )


# =============================================================================
# MiddlewareChain Tests with Real Plugins
# =============================================================================


def test_empty_chain_runs_without_error_with_real_plugin(tmp_path: Path) -> None:
    """Empty chain should not fail with real RepoScanPlugin."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        chain = MiddlewareChain(middleware=[])
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        # None of these should raise
        chain.run_before(plugin, ctx)
        chain.run_after(plugin, ctx, result)
        chain.run_on_error(plugin, ctx, ValueError("test"))
    finally:
        gateway.close()


def test_run_before_calls_all_middleware_with_real_plugin(tmp_path: Path) -> None:
    """run_before should call all middleware in order with real plugin."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        mw1 = RecordingMiddleware()
        mw2 = RecordingMiddleware()
        chain = MiddlewareChain(middleware=[mw1, mw2])
        plugin = RepoScanPlugin()

        chain.run_before(plugin, ctx)

        assert mw1.calls == [("before", "repo_scan")]
        assert mw2.calls == [("before", "repo_scan")]
    finally:
        gateway.close()


def test_run_after_calls_all_middleware_with_real_plugin(tmp_path: Path) -> None:
    """run_after should call all middleware in order with real plugin."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        mw1 = RecordingMiddleware()
        mw2 = RecordingMiddleware()
        chain = MiddlewareChain(middleware=[mw1, mw2])
        plugin = AstExtractPlugin()
        result = IngestPluginResult.ok(row_counts={"core.ast_nodes": 5})

        chain.run_after(plugin, ctx, result)

        assert mw1.calls == [("after", "ast_extract")]
        assert mw2.calls == [("after", "ast_extract")]
    finally:
        gateway.close()


def test_run_on_error_calls_all_middleware_with_real_plugin(tmp_path: Path) -> None:
    """run_on_error should call all middleware in order with real plugin."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        mw1 = RecordingMiddleware()
        mw2 = RecordingMiddleware()
        chain = MiddlewareChain(middleware=[mw1, mw2])
        plugin = RepoScanPlugin()
        error = ValueError("test error")

        chain.run_on_error(plugin, ctx, error)

        assert mw1.calls == [("error", "repo_scan")]
        assert mw2.calls == [("error", "repo_scan")]
    finally:
        gateway.close()


def test_middleware_failure_does_not_break_chain(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Middleware failure should be logged but not stop other middleware."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        failing_mw = RecordingMiddleware(should_fail=True)
        recording_mw = RecordingMiddleware()
        chain = MiddlewareChain(middleware=[failing_mw, recording_mw])
        plugin = RepoScanPlugin()

        with caplog.at_level(logging.WARNING):
            chain.run_before(plugin, ctx)

        # Second middleware should still be called
        assert recording_mw.calls == [("before", "repo_scan")]
        # Warning should be logged
        assert "Middleware before_execute failed" in caplog.text
    finally:
        gateway.close()


def test_middleware_failure_logged_for_after_execute(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Middleware failure in after_execute should be logged."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        failing_mw = RecordingMiddleware(should_fail=True)
        chain = MiddlewareChain(middleware=[failing_mw])
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        with caplog.at_level(logging.WARNING):
            chain.run_after(plugin, ctx, result)

        assert "Middleware after_execute failed" in caplog.text
    finally:
        gateway.close()


def test_middleware_failure_logged_for_on_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Middleware failure in on_error should be logged."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        failing_mw = RecordingMiddleware(should_fail=True)
        chain = MiddlewareChain(middleware=[failing_mw])
        plugin = RepoScanPlugin()

        with caplog.at_level(logging.WARNING):
            chain.run_on_error(plugin, ctx, ValueError("test"))

        assert "Middleware on_error failed" in caplog.text
    finally:
        gateway.close()


# =============================================================================
# LoggingMiddleware Tests with Real Plugins
# =============================================================================


def test_logging_before_execute_logs_start_with_real_plugin(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """before_execute should log plugin start with real plugin metadata."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = RepoScanPlugin()

        with caplog.at_level(logging.INFO):
            middleware.before_execute(plugin, ctx)

        assert "Plugin started" in caplog.text
        assert "repo_scan" in caplog.text
        assert "test/repo" in caplog.text
    finally:
        gateway.close()


def test_logging_after_execute_logs_success_with_real_plugin(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """after_execute should log successful completion with row counts."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok(row_counts={"core.modules": EXPECTED_ROW_COUNT_SINGLE})

        middleware.before_execute(plugin, ctx)  # Set start time
        with caplog.at_level(logging.INFO):
            middleware.after_execute(plugin, ctx, result)

        assert "Plugin completed" in caplog.text
        assert "repo_scan" in caplog.text
        assert f"total_rows={EXPECTED_ROW_COUNT_SINGLE}" in caplog.text
    finally:
        gateway.close()


def test_logging_after_execute_logs_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """after_execute should log when plugin is skipped."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = AstExtractPlugin()
        result = IngestPluginResult.skip("No modules to process")

        middleware.before_execute(plugin, ctx)
        with caplog.at_level(logging.INFO):
            middleware.after_execute(plugin, ctx, result)

        assert "Plugin skipped" in caplog.text
        assert "No modules to process" in caplog.text
    finally:
        gateway.close()


def test_logging_after_execute_logs_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """after_execute should log failures at error level."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = RepoScanPlugin()
        result = IngestPluginResult.fail("Something went wrong", error_kind="ValueError")

        middleware.before_execute(plugin, ctx)
        with caplog.at_level(logging.ERROR):
            middleware.after_execute(plugin, ctx, result)

        assert "Plugin failed" in caplog.text
        assert "Something went wrong" in caplog.text
    finally:
        gateway.close()


def test_logging_on_error_logs_exception(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """on_error should log exception details."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = RepoScanPlugin()
        error = ValueError("Test exception")

        middleware.before_execute(plugin, ctx)
        with caplog.at_level(logging.ERROR):
            middleware.on_error(plugin, ctx, error)

        assert "Plugin error" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test exception" in caplog.text
    finally:
        gateway.close()


def test_logging_without_start_time_handles_gracefully(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Methods should handle missing start time gracefully."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware()
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        # Don't call before_execute
        with caplog.at_level(logging.INFO):
            middleware.after_execute(plugin, ctx, result)

        # Should still log, just with 0 duration
        assert "Plugin completed" in caplog.text
    finally:
        gateway.close()


def test_logging_custom_logger_name(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Custom logger name should be respected."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = LoggingMiddleware(logger_name="custom.logger")
        plugin = RepoScanPlugin()

        with caplog.at_level(logging.INFO, logger="custom.logger"):
            middleware.before_execute(plugin, ctx)

        assert any(rec.name == "custom.logger" for rec in caplog.records)
    finally:
        gateway.close()


# =============================================================================
# MetricsMiddleware Tests with Real Plugins
# =============================================================================


def test_metrics_records_duration_on_success(tmp_path: Path) -> None:
    """Duration should be recorded on successful execution."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        in_memory = InMemoryMetrics()
        middleware = MetricsMiddleware(in_memory=in_memory)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok(row_counts={"core.modules": 10})

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        assert len(in_memory.durations) == 1
        duration, attrs = in_memory.durations[0]
        assert duration >= 0
        assert attrs["plugin"] == "repo_scan"
        assert attrs["status"] == "success"
    finally:
        gateway.close()


def test_metrics_records_row_counts(tmp_path: Path) -> None:
    """Row counts should be recorded on successful execution."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        in_memory = InMemoryMetrics()
        middleware = MetricsMiddleware(in_memory=in_memory)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok(row_counts={"core.modules": 42, "core.repo_map": 8})

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        assert len(in_memory.row_counts) == 1
        count, attrs = in_memory.row_counts[0]
        assert count == EXPECTED_ROW_COUNT_SUM  # Sum of all row counts (42 + 8)
        assert attrs["plugin"] == "repo_scan"
    finally:
        gateway.close()


def test_metrics_records_skipped_status(tmp_path: Path) -> None:
    """Skipped status should be recorded."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        in_memory = InMemoryMetrics()
        middleware = MetricsMiddleware(in_memory=in_memory)
        plugin = AstExtractPlugin()
        result = IngestPluginResult.skip("No data")

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        assert len(in_memory.durations) == 1
        _, attrs = in_memory.durations[0]
        assert attrs["status"] == "skipped"
        # No row counts for skipped
        assert len(in_memory.row_counts) == 0
    finally:
        gateway.close()


def test_metrics_records_error_on_failure(tmp_path: Path) -> None:
    """Error should be recorded on execution failure."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        in_memory = InMemoryMetrics()
        middleware = MetricsMiddleware(in_memory=in_memory)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.fail("Oops")

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        assert len(in_memory.durations) == 1
        _, attrs = in_memory.durations[0]
        assert attrs["status"] == "error"
    finally:
        gateway.close()


def test_metrics_on_error_records_metrics(tmp_path: Path) -> None:
    """on_error should record duration and error count."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        in_memory = InMemoryMetrics()
        middleware = MetricsMiddleware(in_memory=in_memory)
        plugin = RepoScanPlugin()
        error = ValueError("Test error")

        middleware.before_execute(plugin, ctx)
        middleware.on_error(plugin, ctx, error)

        assert len(in_memory.durations) == 1
        _, attrs = in_memory.durations[0]
        assert attrs["status"] == "error"
        assert attrs["error_type"] == "ValueError"

        assert len(in_memory.error_counts) == 1
        count, _ = in_memory.error_counts[0]
        assert count == 1
    finally:
        gateway.close()


def test_in_memory_metrics_clear_resets_metrics() -> None:
    """InMemoryMetrics.clear should reset all metrics."""
    in_memory = InMemoryMetrics()
    in_memory.record_duration(1.0, {"test": "value"})
    in_memory.record_rows(10, {"test": "value"})
    in_memory.record_error(1, {"test": "value"})

    in_memory.clear()

    assert len(in_memory.durations) == 0
    assert len(in_memory.row_counts) == 0
    assert len(in_memory.error_counts) == 0


# =============================================================================
# TracingMiddleware Tests with Real Plugins
# =============================================================================


def test_tracing_creates_span_on_execute(tmp_path: Path) -> None:
    """Span should be created for plugin execution."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok(row_counts={"core.modules": 10})

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert "repo_scan" in span.name
        assert span.attributes["plugin.name"] == "repo_scan"
        assert span.attributes["repo"] == "test/repo"
        assert span.ended
    finally:
        gateway.close()


def test_tracing_span_attributes_on_success(tmp_path: Path) -> None:
    """Span should have success attributes."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok(row_counts={"core.modules": EXPECTED_ROW_COUNT_SINGLE})

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        span = tracer.spans[0]
        assert span.attributes["result.success"] is True
        assert span.attributes["result.skipped"] is False
        assert span.attributes["result.total_rows"] == EXPECTED_ROW_COUNT_SINGLE
        assert span.status == "ok"
    finally:
        gateway.close()


def test_tracing_span_attributes_on_skip(tmp_path: Path) -> None:
    """Span should have skip attributes."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = AstExtractPlugin()
        result = IngestPluginResult.skip("No data available")

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        span = tracer.spans[0]
        assert span.attributes["result.skipped"] is True
        assert span.attributes["result.skip_reason"] == "No data available"
    finally:
        gateway.close()


def test_tracing_span_attributes_on_error_result(tmp_path: Path) -> None:
    """Span should have error attributes from result."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.fail("Something broke", error_kind="ValueError")

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        span = tracer.spans[0]
        assert span.attributes["result.success"] is False
        assert span.attributes["result.error"] == "Something broke"
        assert span.attributes["result.error_kind"] == "ValueError"
        assert span.status == "error"
    finally:
        gateway.close()


def test_tracing_on_error_records_exception(tmp_path: Path) -> None:
    """on_error should record exception on span."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = RepoScanPlugin()
        error = ValueError("Test exception")

        middleware.before_execute(plugin, ctx)
        middleware.on_error(plugin, ctx, error)

        span = tracer.spans[0]
        assert span.exception is error
        assert span.status == "error"
        assert span.ended
    finally:
        gateway.close()


def test_tracing_no_tracer_configured(tmp_path: Path) -> None:
    """Should handle gracefully when no tracer is configured."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        middleware = TracingMiddleware()  # No tracer
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        # Should not raise
        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)
    finally:
        gateway.close()


def test_tracing_after_without_before_handles_gracefully(tmp_path: Path) -> None:
    """after_execute without before_execute should handle gracefully."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer)
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        # Don't call before_execute
        middleware.after_execute(plugin, ctx, result)

        # No span should be created
        assert len(tracer.spans) == 0
    finally:
        gateway.close()


def test_tracing_custom_span_name_prefix(tmp_path: Path) -> None:
    """Custom span name prefix should be used."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("test")
        tracer = InMemoryTracer()
        middleware = TracingMiddleware(in_memory=tracer, span_name_prefix="custom.ingest")
        plugin = RepoScanPlugin()
        result = IngestPluginResult.ok()

        middleware.before_execute(plugin, ctx)
        middleware.after_execute(plugin, ctx, result)

        span = tracer.spans[0]
        assert span.name.startswith("custom.ingest")
    finally:
        gateway.close()


def test_in_memory_tracer_clear() -> None:
    """InMemoryTracer.clear should reset spans."""
    tracer = InMemoryTracer()
    tracer.start_span("test", {"attr": "value"})

    tracer.clear()

    assert len(tracer.spans) == 0


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


@pytest.mark.parametrize(
    "middleware_class",
    [
        LoggingMiddleware,
        MetricsMiddleware,
        TracingMiddleware,
    ],
)
def test_middleware_implements_protocol(middleware_class: type) -> None:
    """All middleware classes should implement IngestMiddleware protocol."""
    middleware = middleware_class()
    assert isinstance(middleware, IngestMiddleware)


def test_recording_middleware_implements_protocol() -> None:
    """RecordingMiddleware should implement IngestMiddleware protocol."""
    middleware = RecordingMiddleware()
    assert isinstance(middleware, IngestMiddleware)


# =============================================================================
# Integration Tests with Provisioned Repository
# =============================================================================


def test_middleware_chain_with_real_plugin_execution(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Test middleware chain with real plugin execution flow."""
    mw1 = RecordingMiddleware()
    metrics = InMemoryMetrics()
    tracer = InMemoryTracer()

    chain = MiddlewareChain(
        middleware=[
            mw1,
            LoggingMiddleware(),
            MetricsMiddleware(in_memory=metrics),
            TracingMiddleware(in_memory=tracer),
        ]
    )

    setup = IngestTestSetup.from_repo(
        provisioned_repo.repo_root,
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    ctx = setup.build_context("integration_test")
    plugin = RepoScanPlugin()

    # Simulate full lifecycle
    chain.run_before(plugin, ctx)
    result = IngestPluginResult.ok(row_counts={"core.modules": 2, "core.repo_map": 1})
    chain.run_after(plugin, ctx, result)

    # Verify all middleware was called
    assert ("before", "repo_scan") in mw1.calls
    assert ("after", "repo_scan") in mw1.calls

    # Verify metrics recorded
    assert len(metrics.durations) == 1
    assert len(metrics.row_counts) == 1
    total_rows = metrics.row_counts[0][0]
    expected_row_count = 3
    assert total_rows == expected_row_count

    # Verify trace spans
    assert len(tracer.spans) == 1
    assert tracer.spans[0].attributes["plugin.name"] == "repo_scan"


def test_middleware_chain_error_handling_integration(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Test middleware chain error handling with real plugin."""
    error_recorder = RecordingMiddleware()
    metrics = InMemoryMetrics()

    chain = MiddlewareChain(
        middleware=[
            error_recorder,
            MetricsMiddleware(in_memory=metrics),
        ]
    )

    setup = IngestTestSetup.from_repo(
        provisioned_repo.repo_root,
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    ctx = setup.build_context("error_test")
    plugin = RepoScanPlugin()

    # Simulate error scenario
    chain.run_before(plugin, ctx)
    test_error = RuntimeError("Test pipeline error")
    chain.run_on_error(plugin, ctx, test_error)

    # Verify error was recorded
    assert ("before", "repo_scan") in error_recorder.calls
    assert ("error", "repo_scan") in error_recorder.calls

    # Verify metrics captured error
    assert len(metrics.error_counts) == 1
