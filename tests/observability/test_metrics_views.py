"""Metric view and exemplar filter tests."""

from __future__ import annotations

import pytest

from codeintel.core.config.settings import MetricViewSettings
from codeintel.observability.otel import ObservabilityConfig, _build_exemplar_filter, _build_views

pytest.importorskip("opentelemetry.sdk.metrics")


def test_build_views_emits_expected_instruments() -> None:
    """Ensure metric view construction defines the expected instruments."""
    config = ObservabilityConfig(
        metric_views=MetricViewSettings(
            operation_duration_ms_buckets=(1.0, 2.0),
            grpc_duration_s_buckets=(0.1, 0.5),
        )
    )
    views = _build_views(config)
    names = {getattr(view, "instrument_name", None) for view in views}
    assert "codeintel.operation.duration_ms" in names
    assert "grpc.client.call.duration" in names
    assert "grpc.server.call.duration" in names


def test_build_exemplar_filter_trace_based() -> None:
    """Ensure trace-based exemplar filter is selected when configured."""
    config = ObservabilityConfig(metrics_exemplar_filter="trace_based")
    exemplar = _build_exemplar_filter(config)
    assert exemplar is not None
    assert exemplar.__class__.__name__ == "TraceBasedExemplarFilter"
