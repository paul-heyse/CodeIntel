"""Test-time observability policy helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Literal

from codeintel.core.config.settings import ObservabilitySettings, OtlpExporterSettings

LOG = logging.getLogger(__name__)

TestTelemetryMode = Literal["disabled", "in_memory", "collector_local", "inherit"]

_TEST_MODE_ENV = "CODEINTEL_TEST_TELEMETRY_MODE"
_PYTEST_ENV = "PYTEST_CURRENT_TEST"
_LOCAL_OTLP_ENDPOINT = "http://localhost:4317"


def resolve_test_telemetry_mode() -> TestTelemetryMode | None:
    """Resolve test telemetry mode from environment.

    Returns
    -------
    TestTelemetryMode | None
        Selected test telemetry mode, or None when not running tests.
    """
    raw = os.environ.get(_TEST_MODE_ENV)
    if raw:
        normalized = raw.strip().lower()
        mapping: dict[str, TestTelemetryMode] = {
            "disabled": "disabled",
            "off": "disabled",
            "false": "disabled",
            "in_memory": "in_memory",
            "in-memory": "in_memory",
            "memory": "in_memory",
            "collector_local": "collector_local",
            "collector-local": "collector_local",
            "collector": "collector_local",
            "local": "collector_local",
            "inherit": "inherit",
            "default": "inherit",
            "production": "inherit",
        }
        mode = mapping.get(normalized)
        if mode is not None:
            return mode
        LOG.warning("Unsupported %s value %s; ignoring", _TEST_MODE_ENV, raw)

    if os.environ.get(_PYTEST_ENV):
        return "in_memory"
    return None


def is_test_telemetry_active() -> bool:
    """Return True when test telemetry mode overrides are active.

    Returns
    -------
    bool
        True when a test telemetry override is active.
    """
    mode = resolve_test_telemetry_mode()
    return mode is not None and mode != "inherit"


def should_shutdown_observability_per_command() -> bool:
    """Return True when per-command shutdown is allowed.

    Returns
    -------
    bool
        True when per-command shutdown is permitted.
    """
    mode = resolve_test_telemetry_mode()
    return mode is None or mode == "inherit"


def apply_test_telemetry_settings(
    settings: ObservabilitySettings,
) -> ObservabilitySettings:
    """Apply test telemetry policy to observability settings.

    Parameters
    ----------
    settings
        Base observability settings to override for tests.

    Returns
    -------
    ObservabilitySettings
        Settings with test telemetry policy applied.
    """
    mode = resolve_test_telemetry_mode()
    if mode is None or mode == "inherit":
        return settings

    disabled_grpc = replace(settings.grpc_observability, enabled=False)
    disabled_tracker = replace(settings.hamilton_tracker, enabled=False)

    if mode == "disabled":
        return replace(
            settings,
            enabled=False,
            export_traces=False,
            export_metrics=False,
            export_logs=False,
            console_export=False,
            prometheus_enabled=False,
            logs_auto_instrument=False,
            log_correlation=False,
            logs_trace_filter=False,
            teardown_enabled=False,
            cli_enabled=False,
            grpc_observability=disabled_grpc,
            hamilton_tracker=disabled_tracker,
        )

    if mode == "in_memory":
        return replace(
            settings,
            enabled=True,
            export_traces=False,
            export_metrics=False,
            export_logs=False,
            console_export=False,
            prometheus_enabled=False,
            logs_auto_instrument=False,
            log_correlation=False,
            logs_trace_filter=False,
            grpc_observability=disabled_grpc,
            hamilton_tracker=disabled_tracker,
        )

    if mode == "collector_local":
        local_otlp = _with_local_otlp(settings.otlp)
        local_traces = _with_local_otlp(settings.otlp_traces)
        local_metrics = _with_local_otlp(settings.otlp_metrics)
        local_logs = _with_local_otlp(settings.otlp_logs)
        return replace(
            settings,
            enabled=True,
            otlp=local_otlp,
            otlp_traces=local_traces,
            otlp_metrics=local_metrics,
            otlp_logs=local_logs,
            console_export=False,
            prometheus_enabled=False,
            grpc_observability=disabled_grpc,
            hamilton_tracker=disabled_tracker,
        )

    return settings


def _with_local_otlp(settings: OtlpExporterSettings) -> OtlpExporterSettings:
    return replace(settings, endpoint=_LOCAL_OTLP_ENDPOINT, protocol="grpc")


__all__ = [
    "TestTelemetryMode",
    "apply_test_telemetry_settings",
    "is_test_telemetry_active",
    "resolve_test_telemetry_mode",
    "should_shutdown_observability_per_command",
]
