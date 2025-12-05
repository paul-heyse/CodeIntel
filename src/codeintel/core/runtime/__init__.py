"""Core runtime utilities for plugin execution.

This package provides shared runtime infrastructure used across
graphs, ingestion, and analytics domains:

- **errors**: Centralized error definitions and catchable exceptions
- **timing**: Duration measurement utilities and context managers
- **telemetry**: OpenTelemetry + Prometheus integration for observability
- **retry**: Tenacity-based retry policies for transient failures
- **validation**: Generic validation finding utilities

Singleton Patterns
------------------
Two singleton patterns are available:

1. **SingletonHolder[T]** (from ``codeintel.core.singleton``):
   Use for registries that need ``reset()`` for testing.

2. **cached_singleton** (from this module):
   Use ``@lru_cache(maxsize=1)`` for simple singletons that don't need reset.
   This is a decorator that wraps a factory function.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

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
    utc_now,
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


def cached_singleton[T](factory: Callable[[], T]) -> Callable[[], T]:
    """Create a cached singleton accessor using lru_cache.

    Use this decorator for simple singletons that don't need reset()
    functionality for testing. For registries that need reset(), use
    SingletonHolder from ``codeintel.core.singleton`` instead.

    Parameters
    ----------
    factory
        Function that creates the singleton instance.

    Returns
    -------
    Callable[[], T]
        Cached version of the factory that returns the same instance.

    Examples
    --------
    >>> @cached_singleton
    ... def get_config() -> Config:
    ...     return Config()
    >>> config1 = get_config()
    >>> config2 = get_config()
    >>> config1 is config2
    True
    """
    return lru_cache(maxsize=1)(factory)


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
    "cached_singleton",
    "cap_findings",
    "filter_by_severity",
    "get_runtime_telemetry",
    "group_findings_by_key",
    "has_error_findings",
    "measure_duration",
    "measure_duration_ms",
    "timed",
    "utc_now",
    "with_retry",
    "with_retry_async",
]
