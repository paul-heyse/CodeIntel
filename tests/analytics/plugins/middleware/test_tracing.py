"""Tests for tracing middleware.

This module tests:
- SpanContext dataclass
- Span class and its methods
- SpanExporter base class
- InMemoryExporter for testing
- TracingMiddleware behavior
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from codeintel.analytics.core.base import BasePlugin
from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import PluginResult, PluginStage, ValidationResult
from codeintel.analytics.plugins.middleware.tracing import (
    InMemoryExporter,
    Span,
    SpanContext,
    SpanExporter,
    TracingMiddleware,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fakes.graph_contexts import create_graph_gateway

# Test constants
TEST_TRACE_ID = "trace123"
TEST_SPAN_ID = "span456"
TEST_PARENT_SPAN_ID = "parent789"
TEST_PLUGIN_NAME = "test.plugin"
TEST_PLUGIN_VERSION = "1.0.0"
TEST_PLUGIN_STAGE: PluginStage = "function"
TEST_RUN_ID = "run123"
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
MINIMUM_DURATION_MS = 0.0
MULTIPLE_SPANS_COUNT = 3


# =============================================================================
# Test Plugin Implementation
# =============================================================================


@dataclass
class TracingTestPlugin(BasePlugin):
    """Minimal plugin implementation for tracing middleware tests.

    Attributes
    ----------
    _should_succeed
        Whether execute should succeed.
    """

    plugin_name: ClassVar[str] = TEST_PLUGIN_NAME
    plugin_description: ClassVar[str] = "Test plugin for tracing"
    plugin_stage: ClassVar[PluginStage] = TEST_PLUGIN_STAGE
    plugin_version: ClassVar[str] = TEST_PLUGIN_VERSION

    _should_succeed: bool = True

    def compute(  # noqa: PLR6301
        self,
        ctx: PluginExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Empty row counts.
        """
        _ = ctx  # Required by interface
        return {}

    def validate_inputs(  # noqa: PLR6301
        self,
        ctx: PluginExecutionContext,
    ) -> ValidationResult:
        """Validate inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Always valid.
        """
        _ = ctx  # Required by interface
        return ValidationResult.success()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_gateway() -> Generator[StorageGateway]:
    """Provide a test gateway that auto-closes.

    Yields
    ------
    StorageGateway
        In-memory gateway with schema applied.
    """
    gateway = create_graph_gateway()
    yield gateway
    gateway.close()


@pytest.fixture
def test_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a test snapshot.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=tmp_path)


def make_context(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    run_id: str | None = TEST_RUN_ID,
) -> PluginExecutionContext:
    """Create a plugin execution context for testing.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    run_id
        Run identifier.

    Returns
    -------
    PluginExecutionContext
        Test execution context.
    """
    return PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        run_id=run_id,
    )


def make_result(*, success: bool = True) -> PluginResult:
    """Create a plugin result for testing.

    Parameters
    ----------
    success
        Whether the result indicates success.

    Returns
    -------
    PluginResult
        Test plugin result.
    """
    if success:
        return PluginResult.ok()
    return PluginResult.fail(error="Test failure")


class TestSpanContext:
    """Tests for SpanContext dataclass."""

    @staticmethod
    def test_creates_span_context() -> None:
        """Verify SpanContext stores all fields."""
        ctx = SpanContext(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            parent_span_id=TEST_PARENT_SPAN_ID,
        )
        assert ctx.trace_id == TEST_TRACE_ID
        assert ctx.span_id == TEST_SPAN_ID
        assert ctx.parent_span_id == TEST_PARENT_SPAN_ID

    @staticmethod
    def test_span_context_allows_none_parent() -> None:
        """Verify SpanContext allows None parent_span_id."""
        ctx = SpanContext(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
            parent_span_id=None,
        )
        assert ctx.parent_span_id is None

    @staticmethod
    def test_span_context_is_frozen() -> None:
        """Verify SpanContext is immutable."""
        ctx = SpanContext(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
        )
        with pytest.raises(AttributeError):
            ctx.trace_id = "new_id"  # type: ignore[misc]


class TestSpan:
    """Tests for Span class."""

    @staticmethod
    def _create_span_context() -> SpanContext:
        """Create a test SpanContext.

        Returns
        -------
        SpanContext
            A test span context for use in tests.
        """
        return SpanContext(
            trace_id=TEST_TRACE_ID,
            span_id=TEST_SPAN_ID,
        )

    def test_creates_span(self) -> None:
        """Verify Span stores all fields."""
        ctx = self._create_span_context()
        start_time = time.perf_counter()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=start_time,
        )
        assert span.name == "test_span"
        assert span.context == ctx
        assert span.start_time == start_time
        assert span.end_time is None
        assert span.status == "ok"
        assert span.attributes == {}

    def test_span_finish_sets_end_time(self) -> None:
        """Verify finish() sets end_time."""
        ctx = self._create_span_context()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=time.perf_counter(),
        )
        assert span.end_time is None

        span.finish()
        assert span.end_time is not None

    def test_span_finish_sets_status(self) -> None:
        """Verify finish() sets status."""
        ctx = self._create_span_context()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=time.perf_counter(),
        )
        span.finish(status="error")
        assert span.status == "error"

    def test_span_duration_ms_returns_zero_when_not_finished(self) -> None:
        """Verify duration_ms returns 0 when not finished."""
        ctx = self._create_span_context()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=time.perf_counter(),
        )
        assert span.duration_ms == MINIMUM_DURATION_MS

    def test_span_duration_ms_returns_positive_when_finished(self) -> None:
        """Verify duration_ms returns positive value when finished."""
        ctx = self._create_span_context()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=time.perf_counter(),
        )
        time.sleep(0.001)  # Sleep 1ms to ensure measurable duration
        span.finish()
        assert span.duration_ms > MINIMUM_DURATION_MS

    def test_span_attributes_can_be_modified(self) -> None:
        """Verify span attributes can be added."""
        ctx = self._create_span_context()
        span = Span(
            name="test_span",
            context=ctx,
            start_time=time.perf_counter(),
            attributes={"key1": "value1"},
        )
        span.attributes["key2"] = "value2"
        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"


class TestSpanExporter:
    """Tests for SpanExporter base class."""

    @staticmethod
    def test_exporter_has_export_method() -> None:
        """Verify SpanExporter has export method."""
        exporter = SpanExporter()
        assert callable(exporter.export)

    @staticmethod
    def test_export_method_accepts_span() -> None:
        """Verify export method can be called with a Span."""
        exporter = SpanExporter()
        ctx = SpanContext(trace_id=TEST_TRACE_ID, span_id=TEST_SPAN_ID)
        span = Span(name="test", context=ctx, start_time=time.perf_counter())
        # Should not raise - base implementation does nothing
        exporter.export(span)


class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    @staticmethod
    def test_exporter_starts_empty() -> None:
        """Verify InMemoryExporter starts with no spans."""
        exporter = InMemoryExporter()
        assert exporter.spans == []

    @staticmethod
    def test_export_stores_span() -> None:
        """Verify export stores span in memory."""
        exporter = InMemoryExporter()
        ctx = SpanContext(trace_id=TEST_TRACE_ID, span_id=TEST_SPAN_ID)
        span = Span(name="test", context=ctx, start_time=time.perf_counter())

        exporter.export(span)
        assert len(exporter.spans) == 1
        assert exporter.spans[0] == span

    @staticmethod
    def test_export_stores_multiple_spans() -> None:
        """Verify export stores multiple spans."""
        exporter = InMemoryExporter()
        ctx = SpanContext(trace_id=TEST_TRACE_ID, span_id=TEST_SPAN_ID)

        for i in range(MULTIPLE_SPANS_COUNT):
            span = Span(name=f"span_{i}", context=ctx, start_time=time.perf_counter())
            exporter.export(span)

        assert len(exporter.spans) == MULTIPLE_SPANS_COUNT

    @staticmethod
    def test_clear_removes_all_spans() -> None:
        """Verify clear removes all stored spans."""
        exporter = InMemoryExporter()
        ctx = SpanContext(trace_id=TEST_TRACE_ID, span_id=TEST_SPAN_ID)

        for i in range(MULTIPLE_SPANS_COUNT):
            span = Span(name=f"span_{i}", context=ctx, start_time=time.perf_counter())
            exporter.export(span)

        exporter.clear()
        assert exporter.spans == []


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    @staticmethod
    def test_middleware_name() -> None:
        """Verify middleware name property."""
        middleware = TracingMiddleware()
        assert middleware.name == "tracing"

    @staticmethod
    def test_before_execute_does_not_export_immediately(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify before_execute does not export immediately."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)

        middleware.before_execute(ctx, plugin)

        # Before after_execute, no spans should be exported yet
        assert len(exporter.spans) == 0

    @staticmethod
    def test_uses_explicit_trace_id(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify middleware uses explicit trace_id when set."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter, trace_id="explicit_trace")
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=True)

        # Execute full lifecycle to get exported span
        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        # Now check the exported span
        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.context.trace_id == "explicit_trace"

    @staticmethod
    def test_uses_run_id_as_trace_id_when_not_set(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify middleware uses run_id as trace_id when trace_id not set."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot, run_id="custom_run_id")
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.context.trace_id == "custom_run_id"

    @staticmethod
    def test_uses_fallback_when_no_run_id(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify middleware uses fallback trace_id when run_id is None."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot, run_id=None)
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.context.trace_id == "no-run-id"

    @staticmethod
    def test_span_has_correct_name(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify exported span has correct name."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.name == f"plugin.{TEST_PLUGIN_NAME}"

    @staticmethod
    def test_span_has_correct_attributes(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify exported span has correct attributes."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.attributes["plugin.name"] == TEST_PLUGIN_NAME
        assert span.attributes["plugin.version"] == TEST_PLUGIN_VERSION
        assert span.attributes["plugin.stage"] == TEST_PLUGIN_STAGE
        assert span.attributes["run.id"] == TEST_RUN_ID
        assert span.attributes["repo"] == TEST_REPO
        assert span.attributes["commit"] == TEST_COMMIT

    @staticmethod
    def test_successful_execution_marks_span_ok(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify successful execution sets span status to ok."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.status == "ok"

    @staticmethod
    def test_failed_execution_marks_span_error(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify failed execution sets span status to error."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=False)

        middleware.before_execute(ctx, plugin)
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.status == "error"

    @staticmethod
    def test_span_has_positive_duration(
        test_gateway: StorageGateway,
        test_snapshot: SnapshotRef,
    ) -> None:
        """Verify finished span has positive duration."""
        exporter = InMemoryExporter()
        middleware = TracingMiddleware(exporter=exporter)
        plugin = TracingTestPlugin()
        ctx = make_context(test_gateway, test_snapshot)
        result = make_result(success=True)

        middleware.before_execute(ctx, plugin)
        time.sleep(0.001)  # Sleep 1ms to ensure measurable duration
        middleware.after_execute(ctx, plugin, result)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.duration_ms > MINIMUM_DURATION_MS
