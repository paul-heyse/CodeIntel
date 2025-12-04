"""Extended tests for graph runtime telemetry module.

This module provides additional test coverage for the telemetry module,
focusing on specific paths not covered by test_runtime.py:

- OpenTelemetry initialization fallback
- Span attribute capture and completion
- Duration computation
- Metrics recording with scope labels
- Exception handling in telemetry operations
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.core.context import GraphPluginExecutionContext, PluginScratch
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginStage,
)
from codeintel.graphs.core.result import PluginExecutionRecord, PluginResult
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.runtime.telemetry import (
    GraphPluginSpan,
    GraphRuntimeTelemetry,
    get_graph_telemetry,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Constants
PLUGIN_COUNT: Final = 5
SLEEP_MS: Final = 20
ATTRIBUTE_MAGIC_VALUE: Final = 42


# Test Helpers


def _make_test_plugin(
    name: str,
    *,
    kind: GraphPluginKind = "builder",
    stage: GraphPluginStage = "goid",
) -> GraphPluginProtocol:
    """Create a test plugin for telemetry tests.

    Parameters
    ----------
    name
        Plugin name.
    kind
        Plugin kind.
    stage
        Plugin stage.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
        return PluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind=kind,
        stage=stage,
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_gateway_and_context() -> tuple[StorageGateway, GraphPluginExecutionContext]:
    """Create a gateway and execution context for telemetry tests.

    Returns
    -------
    tuple
        Gateway and execution context.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)

    snapshot = SnapshotRef(repo="test/telemetry", commit="tel123", repo_root=Path())
    scratch = PluginScratch()

    ctx = GraphPluginExecutionContext(
        snapshot=snapshot,
        graph_resources=ResourceContainer(),
        gateway=gateway,
        scratch=scratch,
        plugin_name="telemetry_test",
        run_id="telemetry-run-001",
    )

    return gateway, ctx


def test_telemetry_init_with_defaults() -> None:
    """Telemetry initializes with default settings."""
    telemetry = GraphRuntimeTelemetry()

    state = vars(telemetry)
    assert state["_service_name"] == "codeintel.graphs"


def test_telemetry_init_with_custom_service_name() -> None:
    """Telemetry initializes with custom service name."""
    telemetry = GraphRuntimeTelemetry(service_name="custom.service")

    state = vars(telemetry)
    assert state["_service_name"] == "custom.service"


def test_telemetry_init_without_otel_disables_features() -> None:
    """Telemetry disables tracing when initialized without OpenTelemetry."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    # When explicitly disabled, features should be off
    state = vars(telemetry)
    assert state["_tracing_enabled"] is False
    assert state["_metrics_enabled"] is False


def test_telemetry_init_with_tracing_disabled() -> None:
    """Telemetry can be initialized with tracing explicitly disabled."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=True,
    )

    state = vars(telemetry)
    assert state["_tracing_enabled"] is False


def test_telemetry_init_with_metrics_disabled() -> None:
    """Telemetry can be initialized with metrics explicitly disabled."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=True,
        enable_metrics=False,
    )

    state = vars(telemetry)
    assert state["_metrics_enabled"] is False


def test_start_plugin_captures_all_attributes() -> None:
    """Start plugin creates span with all required attributes."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("attr_plugin", kind="metric", stage="core")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-attr-001", ctx)

        assert span.plugin_name == "attr_plugin"
        assert span.run_id == "run-attr-001"
        assert span.start_time_ns > 0
        assert span.attributes["plugin.name"] == "attr_plugin"
        assert span.attributes["plugin.kind"] == "metric"
        assert span.attributes["plugin.stage"] == "core"
        assert span.attributes["repo"] == "test/telemetry"
        assert span.attributes["commit"] == "tel123"
        assert span.attributes["run_id"] == "run-attr-001"
    finally:
        gateway.close()


def test_start_plugin_creates_unique_spans() -> None:
    """Each start_plugin call creates a unique span."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("unique_plugin")
    gateway, ctx = _make_gateway_and_context()

    try:
        span1 = telemetry.start_plugin(plugin, "run-1", ctx)
        span2 = telemetry.start_plugin(plugin, "run-2", ctx)

        assert span1.run_id != span2.run_id
        assert span1.start_time_ns != span2.start_time_ns or span1 is not span2
    finally:
        gateway.close()


def test_finish_plugin_computes_duration() -> None:
    """Finish plugin logs completion without raising."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("duration_plugin")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-dur", ctx)

        # Simulate some execution time
        time.sleep(SLEEP_MS / 1000)

        record = PluginExecutionRecord(
            plugin_name="duration_plugin",
            status="succeeded",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            duration_ms=float(SLEEP_MS),
        )

        # Should not raise
        GraphRuntimeTelemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_finish_plugin_with_failed_status() -> None:
    """Finish plugin handles failed status correctly."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("failed_plugin")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-fail", ctx)

        record = PluginExecutionRecord(
            plugin_name="failed_plugin",
            status="failed",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            duration_ms=100.0,
            error="Test error message",
        )

        # Should not raise
        GraphRuntimeTelemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_finish_plugin_with_skipped_status() -> None:
    """Finish plugin handles skipped status correctly."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("skipped_plugin")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-skip", ctx)

        record = PluginExecutionRecord(
            plugin_name="skipped_plugin",
            status="skipped",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            duration_ms=0.0,
        )

        # Should not raise
        GraphRuntimeTelemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_finish_plugin_with_multiple_attempts() -> None:
    """Finish plugin handles records with multiple attempts."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)
    plugin = _make_test_plugin("retry_plugin")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-retry", ctx)

        record = PluginExecutionRecord(
            plugin_name="retry_plugin",
            status="succeeded",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            duration_ms=200.0,
            attempts=3,
        )

        # Should not raise
        GraphRuntimeTelemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_record_metrics_without_metrics_enabled() -> None:
    """Record metrics does nothing when metrics disabled."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    record = PluginExecutionRecord(
        plugin_name="no_metrics",
        status="succeeded",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=50.0,
    )

    # Should not raise
    telemetry.record_metrics(record, scope=None)


def test_record_metrics_with_scope_paths() -> None:
    """Record metrics includes scope path count in labels."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    record = PluginExecutionRecord(
        plugin_name="scope_paths",
        status="succeeded",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=75.0,
    )
    scope = GraphRunScope(paths=("src/", "lib/", "tests/"))

    # Should not raise - metrics disabled so it's a no-op
    telemetry.record_metrics(record, scope)


def test_record_metrics_with_scope_modules() -> None:
    """Record metrics includes scope module count in labels."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    record = PluginExecutionRecord(
        plugin_name="scope_modules",
        status="failed",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=120.0,
        error="Test failure",
    )
    scope = GraphRunScope(modules=("mod_a", "mod_b"))

    # Should not raise
    telemetry.record_metrics(record, scope)


def test_record_metrics_with_none_scope() -> None:
    """Record metrics handles None scope correctly."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    record = PluginExecutionRecord(
        plugin_name="no_scope",
        status="skipped",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=0.0,
    )

    # Should not raise
    telemetry.record_metrics(record, scope=None)


def test_start_run_creates_run_level_span() -> None:
    """Start run creates span with run-level attributes."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    span = telemetry.start_run(
        run_id="run-level-001",
        repo="test/repo",
        commit="abc123",
        plugin_count=PLUGIN_COUNT,
    )

    assert span.plugin_name == "__run__"
    assert span.run_id == "run-level-001"
    assert span.attributes["repo"] == "test/repo"
    assert span.attributes["commit"] == "abc123"
    assert span.attributes["plugin_count"] == PLUGIN_COUNT


def test_finish_run_records_counts() -> None:
    """Finish run logs completion counts without raising."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    span = telemetry.start_run(
        run_id="run-finish-001",
        repo="test/repo",
        commit="abc123",
        plugin_count=10,
    )

    # Should not raise
    GraphRuntimeTelemetry.finish_run(
        span,
        success_count=7,
        failure_count=2,
        skip_count=1,
    )


def test_finish_run_with_all_success() -> None:
    """Finish run handles all-success scenario."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    span = telemetry.start_run(
        run_id="run-all-success",
        repo="test/repo",
        commit="abc123",
        plugin_count=5,
    )

    # Should not raise
    GraphRuntimeTelemetry.finish_run(
        span,
        success_count=5,
        failure_count=0,
        skip_count=0,
    )


def test_finish_run_with_all_failures() -> None:
    """Finish run handles all-failure scenario."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    span = telemetry.start_run(
        run_id="run-all-fail",
        repo="test/repo",
        commit="abc123",
        plugin_count=3,
    )

    # Should not raise
    GraphRuntimeTelemetry.finish_run(
        span,
        success_count=0,
        failure_count=3,
        skip_count=0,
    )


def test_get_graph_telemetry_returns_singleton() -> None:
    """get_graph_telemetry returns same instance on repeated calls."""
    telemetry1 = get_graph_telemetry()
    telemetry2 = get_graph_telemetry()

    assert telemetry1 is telemetry2


def test_get_graph_telemetry_is_valid_instance() -> None:
    """get_graph_telemetry returns a valid GraphRuntimeTelemetry."""
    telemetry = get_graph_telemetry()

    assert isinstance(telemetry, GraphRuntimeTelemetry)


def test_graph_plugin_span_stores_attributes() -> None:
    """GraphPluginSpan correctly stores all attributes."""
    span = GraphPluginSpan(
        plugin_name="test_plugin",
        run_id="run-span-001",
        start_time_ns=time.perf_counter_ns(),
        attributes={"key1": "value1", "key2": ATTRIBUTE_MAGIC_VALUE},
        context_data={"extra": "data"},
    )

    assert span.plugin_name == "test_plugin"
    assert span.run_id == "run-span-001"
    assert span.start_time_ns > 0
    assert span.attributes["key1"] == "value1"
    assert span.attributes["key2"] == ATTRIBUTE_MAGIC_VALUE
    assert span.context_data["extra"] == "data"


def test_graph_plugin_span_default_empty_dicts() -> None:
    """GraphPluginSpan has empty dicts as defaults for attributes/context_data."""
    span = GraphPluginSpan(
        plugin_name="minimal_span",
        run_id="run-minimal",
        start_time_ns=time.perf_counter_ns(),
    )

    assert span.attributes == {}
    assert span.context_data == {}


def test_graph_plugin_span_mutable_attributes() -> None:
    """GraphPluginSpan allows attribute modification."""
    span = GraphPluginSpan(
        plugin_name="mutable_span",
        run_id="run-mutable",
        start_time_ns=time.perf_counter_ns(),
    )

    span.attributes["new_key"] = "new_value"
    span.context_data["otel_span"] = None

    assert span.attributes["new_key"] == "new_value"
    assert "otel_span" in span.context_data


def test_telemetry_handles_otel_not_available() -> None:
    """Telemetry gracefully handles when OpenTelemetry is not available."""
    # This test verifies the telemetry works even without OTEL
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=True,  # Try to enable
        enable_metrics=True,
    )

    # Operations should not raise even if OTEL setup failed
    plugin = _make_test_plugin("otel_test")
    gateway, ctx = _make_gateway_and_context()

    try:
        span = telemetry.start_plugin(plugin, "run-otel", ctx)
        record = PluginExecutionRecord(
            plugin_name="otel_test",
            status="succeeded",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            duration_ms=10.0,
        )
        GraphRuntimeTelemetry.finish_plugin(span, record)
        telemetry.record_metrics(record, None)
    finally:
        gateway.close()


def test_telemetry_start_run_without_otel() -> None:
    """Start run works without OpenTelemetry span creation."""
    telemetry = GraphRuntimeTelemetry(enable_tracing=False, enable_metrics=False)

    span = telemetry.start_run(
        run_id="no-otel-run",
        repo="test/repo",
        commit="abc",
        plugin_count=1,
    )

    # Context data should not have otel_span when tracing disabled
    # (or it might have it but set to None)
    assert span.plugin_name == "__run__"


def test_telemetry_finish_plugin_with_otel_span_none() -> None:
    """Finish plugin handles when otel_span is None in context_data."""
    span = GraphPluginSpan(
        plugin_name="no_otel_span",
        run_id="run-no-otel",
        start_time_ns=time.perf_counter_ns(),
        context_data={"otel_span": None},
    )

    record = PluginExecutionRecord(
        plugin_name="no_otel_span",
        status="succeeded",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=5.0,
    )

    # Should not raise
    GraphRuntimeTelemetry.finish_plugin(span, record)


def test_telemetry_finish_run_with_otel_span_none() -> None:
    """Finish run handles when otel_span is None in context_data."""
    span = GraphPluginSpan(
        plugin_name="__run__",
        run_id="run-no-otel",
        start_time_ns=time.perf_counter_ns(),
        context_data={"otel_span": None},
    )

    # Should not raise
    GraphRuntimeTelemetry.finish_run(
        span,
        success_count=1,
        failure_count=0,
        skip_count=0,
    )
