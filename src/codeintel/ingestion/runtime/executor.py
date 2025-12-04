"""Plugin execution infrastructure for ingestion.

This module provides the low-level execution infrastructure for running
ingestion plugins, including timeout handling, retry logic, and error
classification. Analogous to graphs/runtime/executor.py.

The RecipeExecutor in recipes/executor.py orchestrates high-level recipe
execution and delegates to these components for individual plugin execution.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from codeintel.core.runtime.errors import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
)
from codeintel.ingestion.plugins.protocol import (
    IngestPluginProtocol,
    IngestPluginResult,
)
from codeintel.ingestion.runtime.telemetry import (
    IngestRuntimeTelemetry,
    get_ingest_telemetry,
)
from codeintel.storage.run_tracking import PipelineStatus, StepStatus

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


@dataclass
class PluginExecutionRecord:
    """Record of a single plugin execution.

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    result
        Execution result if successful.
    duration_s
        Execution duration in seconds.
    error
        Exception if execution failed.
    started_at
        Timestamp when execution started.
    ended_at
        Timestamp when execution ended.
    rows_written
        Total rows written across all tables.
    table_counts
        Mapping of table names to row counts.
    """

    plugin_name: str
    result: IngestPluginResult | None = None
    duration_s: float = 0.0
    error: Exception | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    ended_at: datetime | None = None
    rows_written: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if execution succeeded without errors."""
        return self.error is None and self.result is not None

    @property
    def status(self) -> StepStatus:
        """Return the step status for run tracking."""
        if self.error is not None:
            return "failed"
        if self.result is None:
            return "skipped"
        return "succeeded"


@dataclass
class PluginExecutionSettings:
    """Resolved execution settings for a single plugin.

    Attributes
    ----------
    name
        Plugin name.
    severity
        Failure severity level.
    timeout_s
        Execution timeout in seconds.
    fail_fast
        Whether to abort on failure.
    max_retries
        Maximum retry attempts.
    """

    name: str
    severity: PluginSeverity = "soft_fail"
    timeout_s: int | None = None
    fail_fast: bool = True
    max_retries: int = 0


@dataclass
class IngestExecutorConfig:
    """Configuration for plugin execution.

    Attributes
    ----------
    run_id
        Unique run identifier.
    enable_parallel
        Whether to enable parallel execution.
    max_workers
        Maximum thread workers for parallel stages.
    default_timeout_s
        Default timeout per plugin in seconds.
    telemetry
        Telemetry instance for metrics and tracing.
    """

    run_id: str = ""
    enable_parallel: bool = True
    max_workers: int = 4
    default_timeout_s: int | None = None
    telemetry: IngestRuntimeTelemetry = field(default_factory=get_ingest_telemetry)


@dataclass
class IngestRunReport:
    """Report from a batch plugin execution.

    Attributes
    ----------
    run_id
        Run identifier.
    records
        Execution records for each plugin.
    started_at
        Batch start time.
    ended_at
        Batch end time.
    status
        Overall pipeline status.
    """

    run_id: str
    records: list[PluginExecutionRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    ended_at: datetime | None = None
    status: PipelineStatus = "running"

    @property
    def duration_s(self) -> float:
        """Return total duration in seconds."""
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        """Return True if all plugins succeeded."""
        return all(r.success for r in self.records)

    @property
    def total_rows_written(self) -> int:
        """Return total rows written across all plugins."""
        return sum(r.rows_written for r in self.records)

    def get_table_counts(self) -> dict[str, int]:
        """Aggregate table counts from all records.

        Returns
        -------
        dict[str, int]
            Mapping of table names to total row counts.
        """
        result: dict[str, int] = {}
        for record in self.records:
            for table, count in record.table_counts.items():
                result[table] = result.get(table, 0) + count
        return result


def execute_plugin(
    plugin: IngestPluginProtocol,
    context: IngestExecutionContext,
    *,
    settings: PluginExecutionSettings | None = None,
    telemetry: IngestRuntimeTelemetry | None = None,
) -> PluginExecutionRecord:
    """Execute a single plugin and return an execution record.

    Parameters
    ----------
    plugin
        Plugin to execute.
    context
        Execution context with dependencies.
    settings
        Optional execution settings.
    telemetry
        Optional telemetry instance.

    Returns
    -------
    PluginExecutionRecord
        Record of the execution including result or error.
    """
    settings = settings or PluginExecutionSettings(name=plugin.metadata.name)
    telemetry = telemetry or get_ingest_telemetry()

    record = PluginExecutionRecord(
        plugin_name=plugin.metadata.name,
        started_at=datetime.now(tz=UTC),
    )

    span = telemetry.start_plugin_span(plugin, settings.name)

    try:
        result = plugin.execute(context)
        record.result = result
        record.rows_written = sum((result.row_counts or {}).values())
        record.table_counts = dict(result.row_counts or {})
        record.ended_at = datetime.now(tz=UTC)
        record.duration_s = telemetry.end_span(span, success=True, rows_written=record.rows_written)

        log.info(
            "Plugin completed: plugin=%s duration=%.3fs rows=%d",
            plugin.metadata.name,
            record.duration_s,
            record.rows_written,
        )

    except PLUGIN_CATCHABLE_ERRORS as exc:
        record.error = exc
        record.ended_at = datetime.now(tz=UTC)
        record.duration_s = telemetry.end_span(span, success=False, error=str(exc))

        log.warning(
            "Plugin failed: plugin=%s error=%s duration=%.3fs",
            plugin.metadata.name,
            exc,
            record.duration_s,
        )

    return record


def execute_plugin_with_timeout(
    plugin: IngestPluginProtocol,
    context: IngestExecutionContext,
    *,
    timeout_s: int | None = None,
    settings: PluginExecutionSettings | None = None,
    telemetry: IngestRuntimeTelemetry | None = None,
) -> PluginExecutionRecord:
    """Execute a plugin with optional timeout.

    Parameters
    ----------
    plugin
        Plugin to execute.
    context
        Execution context with dependencies.
    timeout_s
        Timeout in seconds (None for no timeout).
    settings
        Optional execution settings.
    telemetry
        Optional telemetry instance.

    Returns
    -------
    PluginExecutionRecord
        Record of the execution including result or error.
    """
    if timeout_s is None:
        return execute_plugin(plugin, context, settings=settings, telemetry=telemetry)

    telemetry = telemetry or get_ingest_telemetry()
    record = PluginExecutionRecord(
        plugin_name=plugin.metadata.name,
        started_at=datetime.now(tz=UTC),
    )

    span = telemetry.start_plugin_span(plugin, plugin.metadata.name)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_plugin, plugin, context, settings=settings, telemetry=None)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            record.error = TimeoutError(f"Plugin timed out after {timeout_s}s")
            record.ended_at = datetime.now(tz=UTC)
            record.duration_s = telemetry.end_span(span, success=False, error="timeout")

            log.warning(
                "Plugin timed out: plugin=%s timeout=%ds",
                plugin.metadata.name,
                timeout_s,
            )

    return record


def execute_plugin_batch(
    plugins: Sequence[IngestPluginProtocol],
    context: IngestExecutionContext,
    *,
    config: IngestExecutorConfig | None = None,
    parallel: bool = False,
) -> IngestRunReport:
    """Execute a batch of plugins sequentially or in parallel.

    Parameters
    ----------
    plugins
        Plugins to execute.
    context
        Execution context with dependencies.
    config
        Executor configuration.
    parallel
        Whether to execute in parallel.

    Returns
    -------
    IngestRunReport
        Report with execution records for each plugin.
    """
    config = config or IngestExecutorConfig()
    report = IngestRunReport(
        run_id=config.run_id or "",
        started_at=datetime.now(tz=UTC),
    )

    if not plugins:
        report.ended_at = datetime.now(tz=UTC)
        report.status = "succeeded"
        return report

    if parallel and config.enable_parallel and len(plugins) > 1:
        report = _execute_batch_parallel(plugins, context, config, report)
    else:
        report = _execute_batch_sequential(plugins, context, config, report)

    report.ended_at = datetime.now(tz=UTC)
    report.status = "succeeded" if report.success else "failed"

    log.info(
        "Batch execution completed: plugins=%d success=%s duration=%.3fs rows=%d",
        len(plugins),
        report.success,
        report.duration_s,
        report.total_rows_written,
    )

    return report


def _execute_batch_sequential(
    plugins: Sequence[IngestPluginProtocol],
    context: IngestExecutionContext,
    config: IngestExecutorConfig,
    report: IngestRunReport,
) -> IngestRunReport:
    """Execute plugins sequentially.

    Parameters
    ----------
    plugins
        Plugins to execute.
    context
        Execution context.
    config
        Executor configuration.
    report
        Report to populate.

    Returns
    -------
    IngestRunReport
        Updated report with execution records.
    """
    for plugin in plugins:
        settings = PluginExecutionSettings(
            name=plugin.metadata.name,
            timeout_s=config.default_timeout_s,
        )
        record = execute_plugin_with_timeout(
            plugin,
            context,
            timeout_s=config.default_timeout_s,
            settings=settings,
            telemetry=config.telemetry,
        )
        report.records.append(record)

    return report


def _execute_batch_parallel(
    plugins: Sequence[IngestPluginProtocol],
    context: IngestExecutionContext,
    config: IngestExecutorConfig,
    report: IngestRunReport,
) -> IngestRunReport:
    """Execute plugins in parallel.

    Parameters
    ----------
    plugins
        Plugins to execute.
    context
        Execution context.
    config
        Executor configuration.
    report
        Report to populate.

    Returns
    -------
    IngestRunReport
        Updated report with execution records.
    """
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                execute_plugin_with_timeout,
                plugin,
                context,
                timeout_s=config.default_timeout_s,
                settings=PluginExecutionSettings(
                    name=plugin.metadata.name,
                    timeout_s=config.default_timeout_s,
                ),
                telemetry=config.telemetry,
            ): plugin
            for plugin in plugins
        }

        for future in futures:
            record = future.result()
            report.records.append(record)

    return report


__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "IngestExecutorConfig",
    "IngestRunReport",
    "PluginExecutionRecord",
    "PluginExecutionSettings",
    "PluginFatalError",
    "PluginSeverity",
    "execute_plugin",
    "execute_plugin_batch",
    "execute_plugin_with_timeout",
]
