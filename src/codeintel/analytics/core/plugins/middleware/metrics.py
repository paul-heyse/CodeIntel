"""Metrics middleware for plugin execution.

This module provides middleware that collects execution metrics
for monitoring and observability.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext
    from codeintel.analytics.core.plugin_protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
    )


@dataclass
class PluginMetrics:
    """Collected metrics for a single plugin execution.

    Attributes
    ----------
    plugin_name
        Name of the plugin.
    duration_ms
        Execution time in milliseconds.
    row_counts
        Row counts per output table.
    success
        Whether execution succeeded.
    error_type
        Error type if failed.
    """

    plugin_name: str
    run_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    duration_ms: float = 0.0
    row_counts: dict[str, int] = field(default_factory=dict)
    success: bool = True
    error_type: str | None = None


@dataclass
class MetricsStore:
    """Store for aggregated plugin metrics.

    Attributes
    ----------
    by_plugin
        Metrics keyed by plugin name.
    by_run
        Metrics keyed by run ID.
    totals
        Aggregate totals.
    """

    by_plugin: dict[str, list[PluginMetrics]] = field(default_factory=lambda: defaultdict(list))
    by_run: dict[str, list[PluginMetrics]] = field(default_factory=lambda: defaultdict(list))
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, run_id: str | None, metrics: PluginMetrics) -> None:
        """Record plugin metrics.

        Parameters
        ----------
        run_id
            Pipeline run ID.
        metrics
            Metrics to record.
        """
        self.by_plugin[metrics.plugin_name].append(metrics)
        self.by_run[run_id or "unknown"].append(metrics)
        self.totals["executions"] += 1
        self.totals["total_duration_ms"] += metrics.duration_ms
        self.totals["total_rows"] += sum(metrics.row_counts.values())
        if not metrics.success:
            self.totals["failures"] += 1

    def get_plugin_stats(self, plugin_name: str) -> dict[str, float]:
        """Get statistics for a plugin.

        Parameters
        ----------
        plugin_name
            Plugin to get stats for.

        Returns
        -------
        dict[str, float]
            Statistics including count, avg_duration, total_rows.
        """
        metrics = self.by_plugin.get(plugin_name, [])
        if not metrics:
            return {"count": 0, "avg_duration_ms": 0, "total_rows": 0}

        total_duration = sum(m.duration_ms for m in metrics)
        total_rows = sum(sum(m.row_counts.values()) for m in metrics)
        failures = sum(1 for m in metrics if not m.success)

        return {
            "count": len(metrics),
            "avg_duration_ms": total_duration / len(metrics),
            "total_duration_ms": total_duration,
            "total_rows": total_rows,
            "failures": failures,
            "success_rate": (len(metrics) - failures) / len(metrics),
        }


# Singleton holder for metrics store
class _MetricsStoreHolder(SingletonHolder["MetricsStore"]):
    """Thread-safe singleton holder for MetricsStore."""


def get_metrics_store() -> MetricsStore:
    """Get or create the global metrics store.

    Returns
    -------
    MetricsStore
        The global store.
    """
    return _MetricsStoreHolder.get(MetricsStore)


def reset_metrics_store() -> None:
    """Reset the global metrics store."""
    _MetricsStoreHolder.reset()


@dataclass
class MetricsMiddleware:
    """Middleware that collects execution metrics.

    Collects timing, row counts, and success/failure rates for
    each plugin execution.

    Attributes
    ----------
    store
        Metrics store to use.
    """

    store: MetricsStore | None = None

    _start_times: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        """Return middleware name."""
        return "metrics"

    def _get_store(self) -> MetricsStore:
        """Get the metrics store to use.

        Returns
        -------
        MetricsStore
            The configured or default metrics store.
        """
        return self.store or get_metrics_store()

    def before_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
    ) -> None:
        """Record execution start time.

        Parameters
        ----------
        ctx
            Execution context (required by interface).
        plugin
            Plugin about to execute.
        """
        plugin_name = plugin.metadata.name
        self._start_times[plugin_name] = time.perf_counter()

    def after_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        result: PluginResult,
    ) -> PluginResult:
        """Record execution metrics.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that executed.
        result
            Execution result.

        Returns
        -------
        PluginResult
            Unchanged result.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0

        metrics = PluginMetrics(
            plugin_name=plugin_name,
            run_id=ctx.run_id,
            repo=ctx.repo,
            commit=ctx.commit,
            duration_ms=duration_ms,
            row_counts=dict(result.row_counts) if result.row_counts else {},
            success=result.success,
            error_type=type(result.error).__name__ if result.error else None,
        )

        self._get_store().record(ctx.run_id, metrics)
        return result

    def on_error(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        error: Exception,
    ) -> Exception | None:
        """Record error metrics.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin that raised.
        error
            The exception raised.

        Returns
        -------
        Exception
            The error unchanged.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0

        metrics = PluginMetrics(
            plugin_name=plugin_name,
            duration_ms=duration_ms,
            success=False,
            error_type=type(error).__name__,
        )

        self._get_store().record(ctx.run_id, metrics)
        return error


__all__ = [
    "MetricsMiddleware",
    "MetricsStore",
    "PluginMetrics",
    "get_metrics_store",
    "reset_metrics_store",
]
