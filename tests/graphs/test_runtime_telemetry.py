"""Tests for graph runtime telemetry.

This module tests the telemetry infrastructure for graph plugin execution
including span management, metric recording, and OpenTelemetry integration.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core.context import GraphExecutionContext, GraphRuntimeScratch
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.result import GraphPluginResult, GraphPluginRunRecord
from codeintel.graphs.runtime.telemetry import (
    GraphPluginSpan,
    GraphRuntimeTelemetry,
    get_graph_telemetry,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

EXPECTED_PLUGIN_COUNT: Final = 5
EXPECTED_ATTR_VALUE: Final = 42


def _make_test_plugin(name: str) -> GraphPluginProtocol:
    """Create a test plugin for telemetry tests.

    Parameters
    ----------
    name
        Plugin name.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_test_context() -> tuple[GraphExecutionContext, object]:
    """Create a test execution context.

    Returns
    -------
    tuple
        A tuple of (GraphExecutionContext, gateway).
        Caller is responsible for closing the gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    scratch = GraphRuntimeScratch()
    ctx = GraphExecutionContext(
        snapshot=snapshot,
        resources=None,
        _gateway=gateway,
        scratch=scratch,
        plugin_name="test_plugin",
        run_id="test-run-123",
    )
    return ctx, gateway


def test_graph_runtime_telemetry_initialization() -> None:
    """Telemetry initializes without raising errors.

    Raises
    ------
    AssertionError
        If telemetry instance is not created.
    """
    # Creating telemetry should not raise
    telemetry = GraphRuntimeTelemetry(
        service_name="test.service",
        enable_tracing=False,
        enable_metrics=False,
    )

    # Verify it's a valid instance
    if telemetry is None:
        msg = "Expected telemetry instance to be created"
        raise AssertionError(msg)


def test_start_plugin_creates_span() -> None:
    """Starting a plugin creates a telemetry span with correct attributes.

    Raises
    ------
    AssertionError
        If span is not created correctly.
    """
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_test_plugin("span_test_plugin")
    ctx, gateway = _make_test_context()

    try:
        span = telemetry.start_plugin(plugin, "run-123", ctx)

        if span.plugin_name != "span_test_plugin":
            msg = f"Expected plugin_name 'span_test_plugin', got '{span.plugin_name}'"
            raise AssertionError(msg)
        if span.run_id != "run-123":
            msg = f"Expected run_id 'run-123', got '{span.run_id}'"
            raise AssertionError(msg)
        if span.start_time_ns <= 0:
            msg = "Expected positive start_time_ns"
            raise AssertionError(msg)
        if span.attributes.get("plugin.name") != "span_test_plugin":
            msg = "Expected plugin.name attribute"
            raise AssertionError(msg)
        if span.attributes.get("repo") != "demo/repo":
            msg = f"Expected repo 'demo/repo', got '{span.attributes.get('repo')}'"
            raise AssertionError(msg)
    finally:
        gateway.close()


def test_finish_plugin_records_duration() -> None:
    """Finishing a plugin span completes without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_test_plugin("duration_test_plugin")
    ctx, gateway = _make_test_context()

    try:
        span = telemetry.start_plugin(plugin, "run-123", ctx)

        # Simulate some execution time
        time.sleep(0.01)

        record = GraphPluginRunRecord(
            name="duration_test_plugin",
            status="succeeded",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=10.0,
        )

        # Should not raise
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_start_run_creates_run_span() -> None:
    """Starting a run creates a run-level telemetry span.

    Raises
    ------
    AssertionError
        If run span is not created correctly.
    """
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    span = telemetry.start_run(
        run_id="run-456",
        repo="demo/repo",
        commit="abc123",
        plugin_count=EXPECTED_PLUGIN_COUNT,
    )

    if span.plugin_name != "__run__":
        msg = f"Expected plugin_name '__run__', got '{span.plugin_name}'"
        raise AssertionError(msg)
    if span.run_id != "run-456":
        msg = f"Expected run_id 'run-456', got '{span.run_id}'"
        raise AssertionError(msg)
    if span.attributes.get("plugin_count") != EXPECTED_PLUGIN_COUNT:
        msg = f"Expected plugin_count {EXPECTED_PLUGIN_COUNT}, got {span.attributes.get('plugin_count')}"
        raise AssertionError(msg)


def test_finish_run_completes_span() -> None:
    """Finishing a run span completes without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    span = telemetry.start_run(
        run_id="run-789",
        repo="demo/repo",
        commit="abc123",
        plugin_count=3,
    )

    # Should not raise
    telemetry.finish_run(span, success_count=2, failure_count=1, skip_count=0)


def test_record_metrics_without_otel() -> None:
    """Recording metrics without OpenTelemetry does not raise."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    record = GraphPluginRunRecord(
        name="metrics_test",
        status="succeeded",
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=datetime.now(tz=UTC).isoformat(),
        duration_ms=100.0,
    )

    # Should not raise even with metrics disabled
    telemetry.record_metrics(record, scope=None)


def test_get_graph_telemetry_singleton() -> None:
    """get_graph_telemetry returns singleton instance.

    Raises
    ------
    AssertionError
        If not a singleton.
    """
    telemetry1 = get_graph_telemetry()
    telemetry2 = get_graph_telemetry()

    if telemetry1 is not telemetry2:
        msg = "Expected get_graph_telemetry to return singleton"
        raise AssertionError(msg)


def test_graph_plugin_span_attributes() -> None:
    """GraphPluginSpan stores attributes correctly.

    Raises
    ------
    AssertionError
        If attributes are not stored correctly.
    """
    span = GraphPluginSpan(
        plugin_name="test_plugin",
        run_id="run-123",
        start_time_ns=time.perf_counter_ns(),
        attributes={"key1": "value1", "key2": EXPECTED_ATTR_VALUE},
        context_data={"otel_span": None},
    )

    if span.plugin_name != "test_plugin":
        msg = f"Expected plugin_name 'test_plugin', got '{span.plugin_name}'"
        raise AssertionError(msg)
    if span.attributes.get("key1") != "value1":
        msg = "Expected key1 attribute to be 'value1'"
        raise AssertionError(msg)
    if span.attributes.get("key2") != EXPECTED_ATTR_VALUE:
        msg = f"Expected key2 attribute to be {EXPECTED_ATTR_VALUE}"
        raise AssertionError(msg)


def test_telemetry_handles_failed_plugin() -> None:
    """Telemetry correctly handles failed plugin records without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_test_plugin("failed_plugin")
    ctx, gateway = _make_test_context()

    try:
        span = telemetry.start_plugin(plugin, "run-fail", ctx)

        record = GraphPluginRunRecord(
            name="failed_plugin",
            status="failed",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=50.0,
            error="Test error message",
        )

        # Should handle failed status without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_telemetry_handles_skipped_plugin() -> None:
    """Telemetry correctly handles skipped plugin records without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_test_plugin("skipped_plugin")
    ctx, gateway = _make_test_context()

    try:
        span = telemetry.start_plugin(plugin, "run-skip", ctx)

        record = GraphPluginRunRecord(
            name="skipped_plugin",
            status="skipped",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=0.0,
        )

        # Should handle skipped status without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_telemetry_with_multiple_attempts() -> None:
    """Telemetry correctly records retry attempts without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_test_plugin("retry_plugin")
    ctx, gateway = _make_test_context()

    try:
        span = telemetry.start_plugin(plugin, "run-retry", ctx)

        record = GraphPluginRunRecord(
            name="retry_plugin",
            status="failed",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=150.0,
            attempts=3,
            error="Failed after retries",
        )

        # Should handle multiple attempts without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()
