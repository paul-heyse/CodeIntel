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
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli_context import make_command_context


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


def test_health_check_handler_output_shape() -> None:
    """Health check handler should return a structured result payload."""
    with make_command_context() as ctx:
        result = health_handler.health_check_handler(ctx)

    expect_true(result.success)
    data = expect_is_not_none(result.data)
    expect_is_instance(data, health_handler.HealthCheckResult)
    expect_true(isinstance(data.checks, list))
    if data.checks:
        first = data.checks[0]
        expect_true(isinstance(first, dict))
        expect_true("name" in first)
        expect_true("status" in first)
        expect_true("message" in first)
        expect_true("duration_ms" in first)
