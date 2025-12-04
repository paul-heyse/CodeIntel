"""Test telemetry infrastructure from codeintel.core.runtime.telemetry.

This module tests:
- PluginSpan dataclass and elapsed properties
- TelemetryConfig configuration
- RuntimeTelemetry initialization
- Graceful fallback when OTEL/Prometheus unavailable
"""

from __future__ import annotations

import time

import pytest

from codeintel.core.runtime.telemetry import (
    DEFAULT_DURATION_BUCKETS,
    OTEL_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    PluginSpan,
    RuntimeTelemetry,
    TelemetryConfig,
    get_runtime_telemetry,
)

# =============================================================================
# PluginSpan Tests
# =============================================================================


def test_plugin_span_construction() -> None:
    """Verify PluginSpan can be constructed with required fields."""
    span = PluginSpan(
        plugin_name="test.plugin",
        run_id="run-123",
        start_time_ns=time.perf_counter_ns(),
    )

    assert span.plugin_name == "test.plugin"
    assert span.run_id == "run-123"
    assert span.start_time_ns > 0


def test_plugin_span_attributes_default_empty() -> None:
    """Verify PluginSpan attributes default to empty dict."""
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=time.perf_counter_ns(),
    )

    assert span.attributes == {}
    assert span.context_data == {}


def test_plugin_span_with_attributes() -> None:
    """Verify PluginSpan accepts attributes."""
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=time.perf_counter_ns(),
        attributes={"key": "value", "count": 42},
    )

    assert span.attributes["key"] == "value"
    assert span.attributes["count"] == 42


def test_plugin_span_elapsed_ns() -> None:
    """Verify elapsed_ns returns nanoseconds."""
    start = time.perf_counter_ns()
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=start,
    )

    time.sleep(0.01)  # 10ms
    elapsed = span.elapsed_ns

    # Should be at least 10ms = 10,000,000 ns
    assert elapsed >= 10_000_000


def test_plugin_span_elapsed_ms() -> None:
    """Verify elapsed_ms returns milliseconds."""
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=time.perf_counter_ns(),
    )

    time.sleep(0.01)  # 10ms
    elapsed = span.elapsed_ms

    # Should be at least 10ms
    assert elapsed >= 10.0


def test_plugin_span_elapsed_s() -> None:
    """Verify elapsed_s returns seconds."""
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=time.perf_counter_ns(),
    )

    time.sleep(0.05)  # 50ms
    elapsed = span.elapsed_s

    # Should be at least 0.05s
    assert elapsed >= 0.05


def test_plugin_span_elapsed_consistency() -> None:
    """Verify elapsed properties are consistent."""
    span = PluginSpan(
        plugin_name="test",
        run_id="run",
        start_time_ns=time.perf_counter_ns(),
    )

    time.sleep(0.01)

    # Get all elapsed values at roughly the same time
    ns = span.elapsed_ns
    ms = span.elapsed_ms
    s = span.elapsed_s

    # Check conversions are approximately correct
    assert abs(ms - ns / 1_000_000) < 1.0  # Within 1ms
    assert abs(s - ns / 1_000_000_000) < 0.001  # Within 1ms


# =============================================================================
# TelemetryConfig Tests
# =============================================================================


def test_telemetry_config_defaults() -> None:
    """Verify TelemetryConfig has sensible defaults."""
    config = TelemetryConfig()

    assert config.service_name == "codeintel"
    assert config.enable_tracing is True
    assert config.enable_metrics is True
    assert config.histogram_buckets == DEFAULT_DURATION_BUCKETS


def test_telemetry_config_custom() -> None:
    """Verify TelemetryConfig accepts custom values."""
    config = TelemetryConfig(
        service_name="my-service",
        enable_tracing=False,
        enable_metrics=False,
        histogram_buckets=(0.1, 0.5, 1.0),
    )

    assert config.service_name == "my-service"
    assert config.enable_tracing is False
    assert config.enable_metrics is False
    assert config.histogram_buckets == (0.1, 0.5, 1.0)


def test_telemetry_config_is_frozen() -> None:
    """Verify TelemetryConfig is immutable."""
    config = TelemetryConfig()

    with pytest.raises(AttributeError):
        config.service_name = "modified"  # type: ignore[misc]


# =============================================================================
# DEFAULT_DURATION_BUCKETS Tests
# =============================================================================


def test_default_duration_buckets() -> None:
    """Verify DEFAULT_DURATION_BUCKETS is a tuple of floats."""
    assert isinstance(DEFAULT_DURATION_BUCKETS, tuple)
    assert all(isinstance(b, float) for b in DEFAULT_DURATION_BUCKETS)


def test_default_duration_buckets_sorted() -> None:
    """Verify DEFAULT_DURATION_BUCKETS are in ascending order."""
    buckets = list(DEFAULT_DURATION_BUCKETS)
    assert buckets == sorted(buckets)


def test_default_duration_buckets_range() -> None:
    """Verify DEFAULT_DURATION_BUCKETS cover a reasonable range."""
    assert min(DEFAULT_DURATION_BUCKETS) > 0  # All positive
    assert min(DEFAULT_DURATION_BUCKETS) < 0.1  # Sub-100ms start
    assert max(DEFAULT_DURATION_BUCKETS) >= 5.0  # At least 5s end


# =============================================================================
# RuntimeTelemetry Tests
# =============================================================================


def test_runtime_telemetry_initialization() -> None:
    """Verify RuntimeTelemetry can be initialized."""
    telemetry = RuntimeTelemetry()

    assert telemetry is not None
    assert telemetry.service_name == "codeintel"


def test_runtime_telemetry_with_config() -> None:
    """Verify RuntimeTelemetry accepts custom config."""
    config = TelemetryConfig(service_name="custom-service")
    telemetry = RuntimeTelemetry(config)

    assert telemetry.service_name == "custom-service"


def test_runtime_telemetry_config_properties() -> None:
    """Verify RuntimeTelemetry exposes config properties."""
    config = TelemetryConfig(enable_tracing=False, enable_metrics=False)
    telemetry = RuntimeTelemetry(config)

    assert telemetry.config_tracing_enabled is False
    assert telemetry.config_metrics_enabled is False


def test_runtime_telemetry_start_span() -> None:
    """Verify RuntimeTelemetry.start_span creates a span."""
    telemetry = RuntimeTelemetry()

    span = telemetry.start_span("test.plugin", "run-123")

    assert isinstance(span, PluginSpan)
    assert span.plugin_name == "test.plugin"
    assert span.run_id == "run-123"


def test_runtime_telemetry_start_span_with_attributes() -> None:
    """Verify RuntimeTelemetry.start_span accepts attributes."""
    telemetry = RuntimeTelemetry()

    span = telemetry.start_span(
        "test.plugin",
        "run-123",
        attributes={"custom": "value"},
    )

    assert span.attributes["custom"] == "value"


def test_runtime_telemetry_end_span_success() -> None:
    """Verify RuntimeTelemetry.end_span records success."""
    telemetry = RuntimeTelemetry()
    span = telemetry.start_span("test.plugin", "run-123")

    time.sleep(0.01)  # Small delay

    duration = telemetry.end_span(span, success=True, rows_written=100)

    assert duration >= 0.01  # At least 10ms


def test_runtime_telemetry_end_span_failure() -> None:
    """Verify RuntimeTelemetry.end_span records failure."""
    telemetry = RuntimeTelemetry()
    span = telemetry.start_span("test.plugin", "run-123")

    duration = telemetry.end_span(
        span,
        success=False,
        error="Something went wrong",
    )

    assert duration >= 0


def test_runtime_telemetry_end_span_returns_duration() -> None:
    """Verify RuntimeTelemetry.end_span returns duration in seconds."""
    telemetry = RuntimeTelemetry()
    span = telemetry.start_span("test.plugin", "run-123")

    time.sleep(0.02)  # 20ms

    duration = telemetry.end_span(span, success=True)

    assert duration >= 0.02  # At least 20ms in seconds


def test_runtime_telemetry_record_run_metrics() -> None:
    """Verify RuntimeTelemetry.record_run_metrics logs metrics."""
    # This is a static method that just logs
    RuntimeTelemetry.record_run_metrics(
        run_id="run-test",
        success_count=10,
        failure_count=2,
        skip_count=1,
        duration_s=5.5,
    )
    # Should not raise


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


def test_otel_available_is_bool() -> None:
    """Verify OTEL_AVAILABLE is a boolean."""
    assert isinstance(OTEL_AVAILABLE, bool)


def test_prometheus_available_is_bool() -> None:
    """Verify PROMETHEUS_AVAILABLE is a boolean."""
    assert isinstance(PROMETHEUS_AVAILABLE, bool)


def test_telemetry_works_without_otel() -> None:
    """Verify telemetry works even when OTEL is unavailable."""
    # Create telemetry - should work regardless of OTEL availability
    telemetry = RuntimeTelemetry()

    span = telemetry.start_span("test", "run")
    duration = telemetry.end_span(span, success=True)

    assert duration >= 0


def test_telemetry_disabled_tracing() -> None:
    """Verify telemetry works with tracing disabled."""
    config = TelemetryConfig(enable_tracing=False)
    telemetry = RuntimeTelemetry(config)

    span = telemetry.start_span("test", "run")
    duration = telemetry.end_span(span, success=True)

    assert duration >= 0
    # OTel span should not be in context_data when tracing disabled
    # (unless OTEL is available and tracer was still created)


def test_telemetry_disabled_metrics() -> None:
    """Verify telemetry works with metrics disabled."""
    config = TelemetryConfig(enable_metrics=False)
    telemetry = RuntimeTelemetry(config)

    span = telemetry.start_span("test", "run")
    duration = telemetry.end_span(span, success=True)

    assert duration >= 0


# =============================================================================
# get_runtime_telemetry Tests
# =============================================================================


def test_get_runtime_telemetry_returns_instance() -> None:
    """Verify get_runtime_telemetry returns a RuntimeTelemetry."""
    telemetry = get_runtime_telemetry()

    assert isinstance(telemetry, RuntimeTelemetry)


def test_get_runtime_telemetry_is_singleton() -> None:
    """Verify get_runtime_telemetry returns the same instance."""
    telemetry1 = get_runtime_telemetry()
    telemetry2 = get_runtime_telemetry()

    assert telemetry1 is telemetry2


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_span_lifecycle() -> None:
    """Verify complete span lifecycle works."""
    telemetry = RuntimeTelemetry()

    # Start span
    span = telemetry.start_span(
        "integration.test",
        "run-456",
        attributes={"test_type": "integration"},
    )

    assert span.plugin_name == "integration.test"
    assert span.run_id == "run-456"
    assert span.attributes["test_type"] == "integration"

    # Do some work
    time.sleep(0.01)

    # End span with success
    duration = telemetry.end_span(span, success=True, rows_written=50)

    assert duration >= 0.01


def test_multiple_spans() -> None:
    """Verify multiple spans can be tracked independently."""
    telemetry = RuntimeTelemetry()

    span1 = telemetry.start_span("plugin1", "run")
    span2 = telemetry.start_span("plugin2", "run")

    time.sleep(0.01)

    duration1 = telemetry.end_span(span1, success=True)
    duration2 = telemetry.end_span(span2, success=False, error="Test error")

    assert duration1 >= 0
    assert duration2 >= 0
    assert span1.plugin_name != span2.plugin_name
