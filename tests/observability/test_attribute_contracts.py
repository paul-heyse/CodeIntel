"""Telemetry attribute registry contract tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from codeintel.observability.attribute_sanitizer import SpanAttributeValue
from codeintel.observability.instrumentation_registry import InstrumentationRegistry
from codeintel.observability.operation_scope import observe_operation
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.runtime import (
    DbTracingConfig,
    MetricConfig,
    ObservabilityConfig,
    ResourceConfig,
    TraceConfig,
    bootstrap_observability,
    shutdown_observability,
)
from codeintel.storage.backend.duckdb_session import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig

trace_api = pytest.importorskip("opentelemetry.trace")
sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
except ImportError:
    pytest.skip("OpenTelemetry metrics SDK unavailable", allow_module_level=True)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class _TelemetryContract(Protocol):
    def assert_valid_attributes(self, attributes: Mapping[str, SpanAttributeValue]) -> None:
        """Assert attributes are valid for the telemetry registry."""


def _configure_tracing(
    *,
    db_tracing: DbTracingConfig | None = None,
) -> InMemorySpanExporter:
    shutdown_observability()
    bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            resources=ResourceConfig(service_name="codeintel-test"),
            traces=TraceConfig(enabled=False, console_export=False),
            metrics=MetricConfig(enabled=False, prometheus_enabled=False),
            db_tracing=db_tracing or DbTracingConfig(),
        )
    )
    exporter = in_memory.InMemorySpanExporter()
    provider = trace_api.get_tracer_provider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    return exporter


def _span_attributes(span: object) -> Mapping[str, SpanAttributeValue]:
    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, Mapping):
        return cast("Mapping[str, SpanAttributeValue]", attributes)
    return {}


def test_operation_span_attributes_registered(telemetry_contract: _TelemetryContract) -> None:
    """Operation span attributes should be registered in the schema."""
    exporter = _configure_tracing()

    with observe_operation(component="cli", operation="health"):
        pass

    spans = exporter.get_finished_spans()
    assert spans
    attrs = _span_attributes(spans[-1])
    telemetry_contract.assert_valid_attributes(attrs)
    shutdown_observability()


def test_db_span_attributes_registered(telemetry_contract: _TelemetryContract) -> None:
    """DB span attributes should be registered in the schema."""
    exporter = _configure_tracing(
        db_tracing=DbTracingConfig(enabled=True, require_parent_span=False),
    )
    session = DuckDBSession(StorageConfig(db_path=Path(":memory:"), repo="r", commit="c"))
    connection = session.open()
    connection.execute("SELECT 1")
    connection.close()

    spans = exporter.get_finished_spans()
    assert spans
    db_spans = [span for span in spans if _span_attributes(span).get("db.system.name") == "duckdb"]
    assert db_spans
    telemetry_contract.assert_valid_attributes(_span_attributes(db_spans[-1]))
    shutdown_observability()


def test_instrumentation_metric_attributes_registered(
    telemetry_contract: _TelemetryContract,
) -> None:
    """Instrumentation metrics should emit registered attribute keys."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("codeintel.test")

    registry = InstrumentationRegistry()
    registry.record_enabled("cli")
    registry.record_error("mcp")
    registry.emit_metrics(meter, policy=ObservabilityPolicy())

    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        pytest.fail("Expected metrics data to be collected")
    found = False
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for point in metric.data.data_points:
                    attrs = point.attributes or {}
                    telemetry_contract.assert_valid_attributes(
                        cast("Mapping[str, SpanAttributeValue]", attrs)
                    )
                    found = True
    assert found
