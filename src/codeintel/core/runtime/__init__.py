"""Core runtime utilities for plugin execution.

This package provides shared runtime infrastructure used across
graphs, ingestion, and analytics domains:

- **errors**: Centralized error definitions and catchable exceptions
- **timing**: Duration measurement utilities and context managers
- **telemetry**: OpenTelemetry + Prometheus integration for observability
- **retry**: Tenacity-based retry policies for transient failures
- **validation**: Generic validation finding utilities
"""

from __future__ import annotations

from codeintel.core.runtime.errors import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginTimeoutError,
)
from codeintel.core.runtime.retry import (
    DATABASE_RETRY_POLICY,
    NETWORK_RETRY_POLICY,
    NO_RETRY_POLICY,
    PLUGIN_RETRY_POLICY,
    RetryError,
    RetryPolicy,
    with_retry,
    with_retry_async,
)
from codeintel.core.runtime.telemetry import (
    DEFAULT_DURATION_BUCKETS,
    OTEL_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    PluginSpan,
    RuntimeTelemetry,
    TelemetryConfig,
    get_runtime_telemetry,
)
from codeintel.core.runtime.timing import (
    TimingResult,
    measure_duration,
    measure_duration_ms,
    timed,
)
from codeintel.core.runtime.validation import (
    BaseValidationOptions,
    SeverityLevel,
    apply_severity_overrides,
    cap_findings,
    filter_by_severity,
    group_findings_by_key,
    has_error_findings,
)

__all__ = [
    "DATABASE_RETRY_POLICY",
    "DEFAULT_DURATION_BUCKETS",
    "NETWORK_RETRY_POLICY",
    "NO_RETRY_POLICY",
    "OTEL_AVAILABLE",
    "PLUGIN_CATCHABLE_ERRORS",
    "PLUGIN_RETRY_POLICY",
    "PROMETHEUS_AVAILABLE",
    "BaseValidationOptions",
    "PluginFatalError",
    "PluginSkippedError",
    "PluginSpan",
    "PluginTimeoutError",
    "RetryError",
    "RetryPolicy",
    "RuntimeTelemetry",
    "SeverityLevel",
    "TelemetryConfig",
    "TimingResult",
    "apply_severity_overrides",
    "cap_findings",
    "filter_by_severity",
    "get_runtime_telemetry",
    "group_findings_by_key",
    "has_error_findings",
    "measure_duration",
    "measure_duration_ms",
    "timed",
    "with_retry",
    "with_retry_async",
]
