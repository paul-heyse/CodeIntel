"""DuckDB tracing smoke tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.observability.otel import (
    ObservabilityConfig,
    bootstrap_observability,
    shutdown_observability,
)
from codeintel.storage.backend.duckdb_session import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

trace_api = pytest.importorskip("opentelemetry.trace")
sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

DIGEST_HEX_LENGTH = 64


def _configure_tracing() -> InMemorySpanExporter:
    """Configure in-memory OpenTelemetry tracing for DuckDB tests.

    Returns
    -------
    InMemorySpanExporter
        Span exporter used to capture emitted spans.
    """
    shutdown_observability()
    bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            otlp_endpoint=None,
            export_traces=False,
            export_metrics=False,
            console_export=False,
            prometheus_enabled=False,
        )
    )

    exporter = in_memory.InMemorySpanExporter()
    provider = trace_api.get_tracer_provider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    return exporter


def _span_attributes(span: object) -> Mapping[str, object]:
    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def test_duckdb_tracing_redacts_statement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure traced statements are redacted."""
    monkeypatch.setenv("CODEINTEL_OTEL_DUCKDB_TRACING", "true")
    monkeypatch.setenv("CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")

    exporter = _configure_tracing()
    session = DuckDBSession(StorageConfig(db_path=Path(":memory:"), repo="r", commit="c"))
    con = session.open()
    con.execute("SELECT 1")
    con.close()

    spans = exporter.get_finished_spans()
    db_spans = [
        span for span in spans if _span_attributes(span).get("db.system.name") == "duckdb"
    ]
    expect_true(bool(db_spans), message="Expected DuckDB spans")

    span = db_spans[-1]
    attrs = _span_attributes(span)
    summary = attrs.get("db.query.summary")
    expect_is_instance(summary, str)
    expect_equal(cast("str", summary), span.name)

    statement = attrs.get("db.statement")
    expect_is_instance(statement, str)
    statement_text = cast("str", statement)
    expect_in("SELECT", statement_text)
    expect_true("1" not in statement_text)

    digest = attrs.get("codeintel.db.statement.sha256")
    expect_is_instance(digest, str)
    digest_text = cast("str", digest)
    expect_equal(len(digest_text), DIGEST_HEX_LENGTH)


def test_duckdb_tracing_operation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure operation-only redaction emits the SQL operation."""
    monkeypatch.setenv("CODEINTEL_OTEL_DUCKDB_TRACING", "true")
    monkeypatch.setenv("CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT", "false")
    monkeypatch.setenv("CODEINTEL_OTEL_DB_STATEMENT_MODE", "operation")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")

    exporter = _configure_tracing()
    session = DuckDBSession(StorageConfig(db_path=Path(":memory:")))
    con = session.open()
    con.execute("SELECT 1")
    con.close()

    spans = exporter.get_finished_spans()
    db_spans = [
        span for span in spans if _span_attributes(span).get("db.system.name") == "duckdb"
    ]
    expect_true(bool(db_spans), message="Expected DuckDB spans")

    attrs = _span_attributes(db_spans[-1])
    summary = attrs.get("db.query.summary")
    expect_is_instance(summary, str)

    statement = attrs.get("db.statement")
    expect_is_instance(statement, str)
    expect_equal(cast("str", statement), "SELECT")
