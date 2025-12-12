"""Unified execution runtime for CodeIntel pipelines.

This package consolidates all runtime execution infrastructure:

Run Context & Identity
----------------------
- **RunContext**: Unified run metadata across all engines
- **RunKind**: Classification of run types (ingest, graphs, analytics, full)
- **TriggerKind**: How the run was triggered (cli, http, mcp, api)
- **new_run_id**: Generate unique run identifiers with prefixes
- **new_run_context**: Factory for creating RunContext instances

Error Handling
--------------
- **PluginFatalError**: Unrecoverable plugin failure
- **PluginSkippedError**: Plugin skipped due to missing prerequisites
- **PluginSkipRequestError**: Internal signal for skip requests
- **PluginTimeoutError**: Plugin execution timeout
- **PLUGIN_CATCHABLE_ERRORS**: Tuple of recoverable plugin errors

Retry & Resilience
------------------
- **RetryPolicy**: Configuration for retry behavior
- **with_retry**: Synchronous retry wrapper using tenacity
- **with_retry_async**: Async retry wrapper
- Predefined policies: PLUGIN_RETRY_POLICY, DATABASE_RETRY_POLICY,
  NETWORK_RETRY_POLICY, NO_RETRY_POLICY

Telemetry & Observability
-------------------------
- **RuntimeTelemetry**: OpenTelemetry + Prometheus integration
- **PluginSpan**: Context manager for plugin execution spans
- **TelemetryConfig**: Configuration for telemetry setup
- **get_runtime_telemetry**: Singleton accessor

Timing Utilities
----------------
- **timed**: Context manager for measuring duration
- **measure_duration**: Wrapper returning TimingResult
- **measure_duration_ms**: Wrapper returning milliseconds
- **utc_now**: UTC-aware datetime.now() helper

Validation
----------
- **BaseValidationOptions**: Configuration for validation passes
- **SeverityLevel**: Warning/error/info classification
- Filtering and grouping utilities for validation findings

Singleton Patterns
------------------
Two patterns are available:

1. **SingletonHolder[T]** (from ``codeintel.core.singleton``):
   Use for registries that need ``reset()`` for testing.

2. **cached_singleton** (from this module):
   Use ``@lru_cache(maxsize=1)`` for simple singletons that don't need reset.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.core.execution.base_context import BaseContext
from codeintel.core.execution.context import RunContext, RunKind, TriggerKind
from codeintel.core.execution.errors import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginSkipRequestError,
    PluginTimeoutError,
)
from codeintel.core.execution.ids import (
    RUN_PREFIX_ANALYTICS,
    RUN_PREFIX_GRAPHS,
    RUN_PREFIX_INGEST,
    RUN_PREFIX_PIPELINE,
    RUN_PREFIX_PLAN,
    new_run_id,
)
from codeintel.core.execution.orchestrator import new_run_context
from codeintel.core.execution.retry import (
    DATABASE_RETRY_POLICY,
    NETWORK_RETRY_POLICY,
    NO_RETRY_POLICY,
    PLUGIN_RETRY_POLICY,
    RetryError,
    RetryPolicy,
    with_retry,
    with_retry_async,
)
from codeintel.core.execution.telemetry import (
    DEFAULT_DURATION_BUCKETS,
    OTEL_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    PluginSpan,
    RuntimeTelemetry,
    TelemetryConfig,
    get_runtime_telemetry,
    reset_runtime_telemetry,
)
from codeintel.core.execution.timing import (
    TimingResult,
    measure_duration,
    measure_duration_ms,
    timed,
    utc_now,
)
from codeintel.core.execution.validation import (
    BaseValidationOptions,
    SeverityLevel,
    apply_severity_overrides,
    cap_findings,
    filter_by_severity,
    group_findings_by_key,
    has_error_findings,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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
    "RUN_PREFIX_ANALYTICS",
    "RUN_PREFIX_GRAPHS",
    "RUN_PREFIX_INGEST",
    "RUN_PREFIX_PIPELINE",
    "RUN_PREFIX_PLAN",
    "BaseContext",
    "BaseValidationOptions",
    "PluginFatalError",
    "PluginSkipRequestError",
    "PluginSkippedError",
    "PluginSpan",
    "PluginTimeoutError",
    "RetryError",
    "RetryPolicy",
    "RunContext",
    "RunKind",
    "RuntimeTelemetry",
    "SeverityLevel",
    "TelemetryConfig",
    "TimingResult",
    "TriggerKind",
    "apply_severity_overrides",
    "cached_singleton",
    "cap_findings",
    "filter_by_severity",
    "get_runtime_telemetry",
    "group_findings_by_key",
    "has_error_findings",
    "measure_duration",
    "measure_duration_ms",
    "new_run_context",
    "new_run_id",
    "reset_runtime_telemetry",
    "timed",
    "utc_now",
    "with_retry",
    "with_retry_async",
]
