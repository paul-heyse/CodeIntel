"""Observability runtime manager tests."""

from __future__ import annotations

import pytest

from codeintel.observability.otel import (
    ObservabilityConfig,
    bootstrap_observability,
    flush_observability,
    get_observability,
    get_pipeline_health_state,
    get_runtime_manager,
    shutdown_observability,
)

pytest.importorskip("opentelemetry.sdk.trace")


def test_runtime_manager_flush_without_bootstrap() -> None:
    """Flush should be a no-op when no runtime is bootstrapped."""
    shutdown_observability()
    get_runtime_manager().reset()
    assert flush_observability() is None


def test_runtime_manager_records_pipeline_health() -> None:
    """Pipeline health should update after a flush."""
    shutdown_observability()
    _ = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            export_traces=False,
            export_metrics=False,
            export_logs=False,
            console_export=False,
            prometheus_enabled=False,
            test_mode="in_memory",
        )
    )
    result = flush_observability()
    assert result is not None
    state = get_pipeline_health_state()
    assert state.last_flush_ok == result.flush_ok
    assert state.last_flush_ms == result.flush_ms
    shutdown_observability()
    assert get_observability().enabled is False
