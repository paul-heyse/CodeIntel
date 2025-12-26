"""Health handler telemetry pipeline tests."""

from __future__ import annotations

from codeintel.cli.handlers import health as health_handler
from codeintel.observability.runtime import (
    LogConfig,
    MetricConfig,
    ObservabilityConfig,
    ResourceConfig,
    TraceConfig,
    bootstrap_observability,
    shutdown_observability,
)
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def _find_check(
    report: health_handler.HealthReport, name: str
) -> health_handler.CheckResult | None:
    for check in report.checks:
        if check.name == name:
            return check
    return None


def test_telemetry_pipeline_check_skips_when_disabled() -> None:
    """Telemetry pipeline check should skip when observability is disabled."""
    shutdown_observability()
    report = health_handler.get_health_checker().run_all()
    result = _find_check(report, "telemetry_pipeline")
    expect_true(result is not None)
    if result is not None:
        expect_equal(result.status, health_handler.CheckStatus.SKIP)


def test_telemetry_pipeline_check_reports_flush() -> None:
    """Telemetry pipeline check should emit a span and flush result."""
    shutdown_observability()
    _ = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            resources=ResourceConfig(service_name="codeintel-test"),
            traces=TraceConfig(enabled=False),
            metrics=MetricConfig(enabled=False),
            logs=LogConfig(enabled=False),
            test_mode="in_memory",
        )
    )
    report = health_handler.get_health_checker().run_all()
    result = _find_check(report, "telemetry_pipeline")
    expect_true(result is not None)
    if result is not None:
        expect_equal(result.status, health_handler.CheckStatus.OK)
        expect_equal(result.name, "telemetry_pipeline")
        expect_true(isinstance(result.details, dict))
    shutdown_observability()
