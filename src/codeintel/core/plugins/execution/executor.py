"""Base plugin executor for unified execution infrastructure.

This module provides the abstract base executor class that unifies
execution logic across analytics, graphs, and ingestion domains.
Domain-specific executors extend this base with their own context
building and report generation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from codeintel.core.execution.errors import PLUGIN_CATCHABLE_ERRORS
from codeintel.core.execution.telemetry import get_runtime_telemetry
from codeintel.core.execution.timing import utc_now
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.execution.tracking import FatalHandling
from codeintel.core.plugins.types.protocol import ValidationResult
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginResult, PluginStatus

if TYPE_CHECKING:
    from codeintel.core.execution.telemetry import RuntimeTelemetry
    from codeintel.core.plugins.execution.context import PluginExecutionContext
    from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
    from codeintel.core.plugins.execution.settings import PluginExecutionSettings
    from codeintel.core.plugins.registry.base import PluginPlan
    from codeintel.core.plugins.types.protocol import PluginProtocol
    from codeintel.core.plugins.types.report import BaseExecutionReport

log = logging.getLogger(__name__)

# Type variables for generic executor
P = TypeVar("P", bound="PluginProtocol")
C = TypeVar("C", bound="PluginExecutionContext")
EC = TypeVar("EC", bound="BaseExecutorContext")
R = TypeVar("R", bound="BaseExecutionReport")


@dataclass(frozen=True)
class ExecutionStrategyContext[P, EC]:
    """Context passed to strategy hooks for plugin execution decisions."""

    plugin: P
    executor_ctx: EC
    policy: BaseExecutionPolicy
    settings: PluginExecutionSettings | None
    prior_manifest: Mapping[str, Mapping[str, object]] | None


class ExecutionStrategy(Enum):
    """Named execution strategies for plugin runs."""

    STANDARD = "standard"


class PluginExecutionStrategy[P, C, EC](Protocol):
    """Strategy interface for execution hooks and manifest building."""

    def should_skip(self, ctx: ExecutionStrategyContext[P, EC]) -> str | None:
        """Return a skip reason or None for the given plugin."""

    def on_success(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        plugin_ctx: C,
        result: PluginResult,
    ) -> None:
        """Handle successful execution."""

    def on_failure(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        plugin_ctx: C,
        error: str | None,
    ) -> None:
        """Handle failed execution."""

    def build_manifest_entry(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        record: PluginExecutionRecord,
    ) -> dict[str, object] | None:
        """Build a manifest entry for a completed plugin."""


class DefaultPluginExecutionStrategy[P, C, EC]:
    """Default execution hooks with dry-run skipping and no manifest tracking."""

    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.STANDARD) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> ExecutionStrategy:
        """Configured execution strategy name."""
        return self._strategy

    def should_skip(self, ctx: ExecutionStrategyContext[P, EC]) -> str | None:
        """Determine whether to skip plugin execution.

        Parameters
        ----------
        ctx
            Strategy context for the plugin run.

        Returns
        -------
        str | None
            Skip reason when skipping, otherwise None.
        """
        # Provide an explicit self reference to allow future strategy-specific branching
        if self._strategy is ExecutionStrategy.STANDARD and ctx.policy.dry_run:
            return "dry_run"
        return None

    def on_success(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        plugin_ctx: C,
        result: PluginResult,
    ) -> None:
        """Invoke the success hook (no-op by default)."""
        _ = (ctx, plugin_ctx, result, self._strategy)

    def on_failure(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        plugin_ctx: C,
        error: str | None,
    ) -> None:
        """Invoke the failure hook (no-op by default)."""
        _ = (ctx, plugin_ctx, error, self._strategy)

    def build_manifest_entry(
        self,
        ctx: ExecutionStrategyContext[P, EC],
        record: PluginExecutionRecord,
    ) -> dict[str, object] | None:
        """Build a manifest entry for tracking if enabled.

        Parameters
        ----------
        ctx
            Strategy context for the plugin run.
        record
            Execution record for the plugin.

        Returns
        -------
        dict[str, object] | None
            Manifest payload, or None when manifest tracking is disabled.
        """
        _ = (ctx, record, self._strategy)
        return None


@dataclass(frozen=True)
class ExecutionOptions[P, C, EC]:
    """Options controlling executor behavior and hook strategy."""

    fatal_handling: FatalHandling = FatalHandling.FAIL_FAST
    strategy: PluginExecutionStrategy[P, C, EC] | None = None
    strategy_name: ExecutionStrategy = ExecutionStrategy.STANDARD

    def __post_init__(self) -> None:
        """Populate missing strategy with the default implementation."""
        if self.strategy is None:
            object.__setattr__(
                self,
                "strategy",
                cast(
                    "PluginExecutionStrategy[P, C, EC]",
                    DefaultPluginExecutionStrategy(),
                ),
            )

    @classmethod
    def from_policy(
        cls,
        policy: BaseExecutionPolicy,
        *,
        strategy: PluginExecutionStrategy[P, C, EC] | None = None,
        strategy_name: ExecutionStrategy = ExecutionStrategy.STANDARD,
    ) -> ExecutionOptions[P, C, EC]:
        """Create execution options derived from an execution policy.

        Returns
        -------
        ExecutionOptions[P, C, EC]
            Options aligned to the provided policy defaults.
        """
        fatal_handling = FatalHandling.FAIL_FAST if policy.fail_fast else FatalHandling.CONTINUE
        return cls(
            fatal_handling=fatal_handling,
            strategy=strategy,
            strategy_name=strategy_name,
        )


@dataclass(frozen=True)
class ExecutionReportContext[EC]:
    """Context for building execution reports."""

    run_id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    fatal_error: bool
    executor_ctx: EC


@dataclass(frozen=True)
class ExecutionTiming:
    """Captured timing data for an individual plugin run."""

    ended_at: datetime
    duration_ms: float


class BasePluginExecutor[
    P: "PluginProtocol",
    C: "PluginExecutionContext",
    EC: "BaseExecutorContext",
    R: "BaseExecutionReport",
](ABC):
    """Abstract base executor with retry, telemetry, and recording.

    Provide unified execution infrastructure for all plugin domains.
    Domain-specific executors extend this base and implement the
    abstract methods for context building and report generation.

    Type Parameters
    ---------------
    P
        Plugin protocol type (e.g., AnalyticsPluginProtocol).
    C
        Plugin execution context type (e.g., PluginExecutionContext).
    EC
        Executor context type (e.g., AnalyticsExecutorContext).
    R
        Execution report type (e.g., AnalyticsExecutionReport).

    Attributes
    ----------
    _policy
        Execution policy controlling behavior.
    _telemetry
        Runtime telemetry for spans and metrics.

    Examples
    --------
    >>> class MyExecutor(BasePluginExecutor[MyPlugin, MyContext, MyExecCtx, MyReport]):
    ...     def _build_plugin_context(self, base_ctx, plugin, scratch):
    ...         return MyContext(...)
    ...
    ...     def _build_report(self, records, report_ctx):
    ...         return MyReport(...)
    """

    def __init__(
        self,
        policy: BaseExecutionPolicy | None = None,
        telemetry: RuntimeTelemetry | None = None,
        prior_manifest: Mapping[str, Mapping[str, object]] | None = None,
        options: ExecutionOptions[P, C, EC] | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        policy
            Execution policy. Uses defaults if not provided.
        telemetry
            Runtime telemetry. Uses default singleton if not provided.
        prior_manifest
            Prior execution manifest for skip detection.
        options
            Execution options such as fatal handling and strategy hooks.
        """
        self._policy = policy or BaseExecutionPolicy()
        self._telemetry = telemetry or get_runtime_telemetry()
        self._prior_manifest = prior_manifest
        self._manifest: dict[str, dict[str, object]] = {}
        self._options = options or ExecutionOptions.from_policy(self._policy)
        self._strategy: PluginExecutionStrategy[P, C, EC] = cast(
            "PluginExecutionStrategy[P, C, EC]",
            self._options.strategy,
        )

    @property
    def policy(self) -> BaseExecutionPolicy:
        """Return the execution policy.

        Returns
        -------
        BaseExecutionPolicy
            Current execution policy.
        """
        return self._policy

    @property
    def telemetry(self) -> RuntimeTelemetry:
        """Return the telemetry instance.

        Returns
        -------
        RuntimeTelemetry
            Current telemetry instance.
        """
        return self._telemetry

    @property
    def options(self) -> ExecutionOptions[P, C, EC]:
        """Return execution options."""
        return self._options

    @abstractmethod
    def _build_plugin_context(
        self,
        executor_ctx: EC,
        plugin: P,
        scratch: PluginScratch,
    ) -> C:
        """Build domain-specific plugin execution context.

        Parameters
        ----------
        executor_ctx
            Executor-level context with gateway, snapshot, etc.
        plugin
            Plugin being executed.
        scratch
            Shared scratch store for inter-plugin communication.

        Returns
        -------
        C
            Plugin execution context for the specific domain.
        """
        ...

    @abstractmethod
    def _build_report(
        self,
        records: list[PluginExecutionRecord],
        report_ctx: ExecutionReportContext[EC],
    ) -> R:
        """Build domain-specific execution report.

        Parameters
        ----------
        records
            Plugin execution records.
        report_ctx
            Context summarizing run identifiers, timing, and execution status.

        Returns
        -------
        R
            Domain-specific execution report.
        """
        ...

    def _should_skip_plugin(self, strategy_ctx: ExecutionStrategyContext[P, EC]) -> str | None:
        """Check if plugin should be skipped using the configured strategy.

        Parameters
        ----------
        strategy_ctx
            Strategy context describing the plugin and executor state.

        Returns
        -------
        str | None
            Skip reason if the plugin should be skipped, otherwise None.
        """
        return self._strategy.should_skip(strategy_ctx)

    def _validate_plugin_inputs(self, plugin: P, ctx: C) -> str | None:
        """Validate plugin inputs before execution.

        Override in subclass to implement input validation.

        Parameters
        ----------
        plugin
            Plugin to validate.
        ctx
            Plugin execution context.

        Returns
        -------
        str | None
            Error message when invalid, otherwise None.
        """
        # Base uses self._policy (even though trivially) to ensure it's an instance method
        _ = self._policy  # Instance method validation hook
        # Check if plugin has validate_inputs method
        validate_method = getattr(plugin, "validate_inputs", None)
        if validate_method is not None and callable(validate_method):
            validation = cast("ValidationResult", validate_method(ctx))
            if not validation.valid:
                return f"Validation failed: {', '.join(validation.errors)}"
        return None

    @staticmethod
    def _get_plugin_settings(
        plugin: P,
        settings_by_plugin: Mapping[str, PluginExecutionSettings] | None,
    ) -> PluginExecutionSettings | None:
        """Get settings for a plugin.

        Parameters
        ----------
        plugin
            Plugin to get settings for.
        settings_by_plugin
            Per-plugin settings map.

        Returns
        -------
        PluginExecutionSettings | None
            Settings if available.
        """
        if settings_by_plugin is None:
            return None
        return settings_by_plugin.get(plugin.metadata.name)

    @property
    def manifest(self) -> dict[str, dict[str, object]]:
        """Return the execution manifest.

        Returns
        -------
        dict[str, dict[str, object]]
            Manifest entries keyed by plugin name.
        """
        return self._manifest

    @property
    def prior_manifest(self) -> Mapping[str, Mapping[str, object]] | None:
        """Return the prior manifest.

        Returns
        -------
        Mapping[str, Mapping[str, object]] | None
            Prior manifest if provided.
        """
        return self._prior_manifest

    def execute_plan(
        self,
        executor_ctx: EC,
        plan: PluginPlan[P],
        *,
        scratch: PluginScratch | None = None,
        run_id: str | None = None,
        settings_by_plugin: Mapping[str, PluginExecutionSettings] | None = None,
    ) -> R:
        """Execute all plugins in plan with retry and telemetry.

        Parameters
        ----------
        executor_ctx
            Executor-level context.
        plan
            Plugin execution plan with ordered plugins.
        scratch
            Optional shared scratch store.
        run_id
            Optional run identifier.
        settings_by_plugin
            Optional per-plugin execution settings.

        Returns
        -------
        R
            Execution report with all plugin results.
        """
        effective_run_id = run_id or executor_ctx.effective_run_id or plan.plan_id
        started_at = utc_now()
        start_time = time.perf_counter()
        records: list[PluginExecutionRecord] = []
        shared_scratch = scratch or PluginScratch()
        fatal_error = False

        # Reset manifest for this run
        self._manifest = {}

        log.info(
            "executor.plan.start run_id=%s plugin_count=%d",
            effective_run_id,
            len(plan.plugins),
        )

        try:
            for plugin in plan.plugins:
                strategy_ctx = ExecutionStrategyContext(
                    plugin=plugin,
                    executor_ctx=executor_ctx,
                    policy=self._policy,
                    settings=self._get_plugin_settings(plugin, settings_by_plugin),
                    prior_manifest=self._prior_manifest,
                )

                # Check for skip condition
                skip_reason = self._should_skip_plugin(strategy_ctx)
                if skip_reason is not None:
                    record = self._create_skip_record(strategy_ctx, skip_reason)
                    records.append(record)
                    continue

                # Build plugin-specific context
                record = self._execute_single_plugin(
                    plugin,
                    self._build_plugin_context(executor_ctx, plugin, shared_scratch),
                    effective_run_id,
                    strategy_ctx=strategy_ctx,
                )
                records.append(record)

                self._maybe_add_manifest_entry(strategy_ctx, record)

                if self._should_stop_on_failure(strategy_ctx, record):
                    log.error(
                        "executor.plan.fatal_error plugin=%s",
                        plugin.metadata.name,
                    )
                    fatal_error = True
                    break
        finally:
            shared_scratch.cleanup()

        run_timing = self._compute_timing(start_time)

        # Record run-level telemetry
        self._telemetry.record_run_metrics(
            run_id=effective_run_id,
            success_count=sum(1 for r in records if r.status == "succeeded"),
            failure_count=sum(1 for r in records if r.status == "failed"),
            skip_count=sum(1 for r in records if r.status == "skipped"),
            duration_s=run_timing.duration_ms / 1000,
        )

        log.info(
            "executor.plan.complete run_id=%s duration_ms=%.2f fatal_error=%s",
            effective_run_id,
            run_timing.duration_ms,
            fatal_error,
        )

        return self._build_report(
            records=records,
            report_ctx=ExecutionReportContext(
                run_id=effective_run_id,
                started_at=started_at,
                ended_at=run_timing.ended_at,
                duration_ms=run_timing.duration_ms,
                fatal_error=fatal_error,
                executor_ctx=executor_ctx,
            ),
        )

    @staticmethod
    def _compute_timing(start_time: float) -> ExecutionTiming:
        """Compute timing information from a start timestamp.

        Returns
        -------
        ExecutionTiming
            Timing data including end timestamp and duration.
        """
        return ExecutionTiming(
            ended_at=utc_now(),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )

    def _execute_single_plugin(
        self,
        plugin: P,
        ctx: C,
        run_id: str,
        *,
        strategy_ctx: ExecutionStrategyContext[P, EC],
    ) -> PluginExecutionRecord:
        """Execute one plugin with retry using core.runtime.retry.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Plugin execution context.
        run_id
            Run identifier for telemetry.
        strategy_ctx
            Strategy context for hooks and policy decisions.

        Returns
        -------
        PluginExecutionRecord
            Execution record with status, duration, and result.
        """
        started_at = utc_now()
        start_time = time.perf_counter()

        # Start telemetry span
        span = self._telemetry.start_span(
            plugin.metadata.name,
            run_id,
            attributes={"stage": plugin.metadata.stage, "kind": plugin.metadata.kind},
        )

        log.info(
            "executor.plugin.start name=%s stage=%s",
            plugin.metadata.name,
            plugin.metadata.stage,
        )

        # Validate inputs
        validation_error = self._validate_plugin_inputs(plugin, ctx)
        if validation_error is not None:
            timing = self._compute_timing(start_time)
            self._telemetry.end_span(span, success=False, error=validation_error)
            return PluginExecutionRecord(
                plugin_name=plugin.metadata.name,
                status="failed",
                started_at=started_at,
                ended_at=timing.ended_at,
                duration_ms=timing.duration_ms,
                error=validation_error,
                meta=self._build_record_meta(strategy_ctx),
            )

        # Execute with retry
        result, attempts, error = self._execute_with_retries(
            plugin,
            ctx,
            strategy_ctx.settings,
        )

        timing = self._compute_timing(start_time)

        status: PluginStatus

        if result is not None and result.success:
            status = "skipped" if result.skipped else "succeeded"
            self._strategy.on_success(strategy_ctx, ctx, result)
            self._telemetry.end_span(
                span,
                success=True,
                rows_written=sum(result.row_counts.values()) if result.row_counts else 0,
            )
        else:
            status = self._failed_status(plugin, strategy_ctx)
            self._strategy.on_failure(strategy_ctx, ctx, error)
            self._telemetry.end_span(span, success=False, error=error)

        log.info(
            "executor.plugin.complete name=%s status=%s duration_ms=%.2f attempts=%d",
            plugin.metadata.name,
            status,
            timing.duration_ms,
            attempts,
        )

        return PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status=status,
            started_at=started_at,
            ended_at=timing.ended_at,
            duration_ms=timing.duration_ms,
            attempts=attempts,
            result=result,
            error=error,
            meta=self._build_record_meta(strategy_ctx),
        )

    def _execute_with_retries(
        self,
        plugin: P,
        ctx: C,
        settings: PluginExecutionSettings | None = None,
    ) -> tuple[PluginResult | None, int, str | None]:
        """Execute plugin with retry logic.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Plugin execution context.
        settings
            Optional per-plugin execution settings.

        Returns
        -------
        tuple[PluginResult | None, int, str | None]
            Result, attempt count, and error message.
        """
        # Determine retry policy:
        # 1. Use settings retry_policy if provided
        # 2. Otherwise use per-plugin override from policy
        if settings is not None:
            retry_policy = settings.retry_policy
        else:
            plugin_name = plugin.metadata.name
            retry_policy = self._policy.get_retry_policy(plugin_name)

        attempts = 0
        error: str | None = None
        result: PluginResult | None = None

        try:
            # Use tenacity-based retry from core.runtime.retry
            for attempt in retry_policy.create_retrying():
                with attempt:
                    attempts += 1
                    result = plugin.execute(ctx)
                    if result.success:
                        return result, attempts, None
                    # Plugin returned failure without exception
                    error = result.error
                    return result, attempts, error
        except PLUGIN_CATCHABLE_ERRORS as exc:
            error = repr(exc)
            log.warning(
                "executor.plugin.error name=%s error=%s",
                plugin.metadata.name,
                error,
            )

        return result, max(attempts, 1), error

    @staticmethod
    def _build_record_meta(strategy_ctx: ExecutionStrategyContext[P, EC]) -> dict[str, object]:
        """Build record metadata from strategy context settings.

        Returns
        -------
        dict[str, object]
            Metadata map containing hashes when available.
        """
        record_meta: dict[str, object] = {}
        settings = strategy_ctx.settings
        if settings is not None:
            record_meta["input_hash"] = settings.input_hash
            record_meta["options_hash"] = settings.options_hash
            record_meta["version_hash"] = settings.version_hash
        return record_meta

    def _maybe_add_manifest_entry(
        self,
        strategy_ctx: ExecutionStrategyContext[P, EC],
        record: PluginExecutionRecord,
    ) -> None:
        """Add a manifest entry when a plugin succeeds."""
        if record.status != "succeeded":
            return
        manifest_entry = self._strategy.build_manifest_entry(strategy_ctx, record)
        if manifest_entry is not None:
            self._manifest[strategy_ctx.plugin.metadata.name] = manifest_entry

    def _should_stop_on_failure(
        self,
        strategy_ctx: ExecutionStrategyContext[P, EC],
        record: PluginExecutionRecord,
    ) -> bool:
        """Decide whether to halt execution based on severity.

        Returns
        -------
        bool
            True when execution should stop, False otherwise.
        """
        if record.status != "failed":
            return False
        severity = (
            strategy_ctx.settings.severity
            if strategy_ctx.settings is not None
            else strategy_ctx.policy.get_severity(strategy_ctx.plugin.metadata.name)
        )
        return severity == "fatal" and self._options.fatal_handling is FatalHandling.FAIL_FAST

    @staticmethod
    def _failed_status(plugin: P, strategy_ctx: ExecutionStrategyContext[P, EC]) -> PluginStatus:
        """Resolve status for a failed plugin based on severity settings.

        Returns
        -------
        PluginStatus
            Status string of either "skipped" or "failed".
        """
        severity = (
            strategy_ctx.settings.severity
            if strategy_ctx.settings is not None
            else strategy_ctx.policy.get_severity(plugin.metadata.name)
        )
        return "skipped" if severity == "skip_on_error" else "failed"

    @staticmethod
    def _create_skip_record(
        strategy_ctx: ExecutionStrategyContext[P, EC],
        reason: str,
    ) -> PluginExecutionRecord:
        """Create a skip record for a plugin.

        Parameters
        ----------
        strategy_ctx
            Strategy context describing the skipped plugin.
        reason
            Reason for skipping.

        Returns
        -------
        PluginExecutionRecord
            Record with skipped status.
        """
        now = utc_now()
        plugin = strategy_ctx.plugin
        settings = strategy_ctx.settings
        record_meta: dict[str, object] = {"skipped_reason": reason}
        if settings is not None:
            record_meta["input_hash"] = settings.input_hash
            record_meta["options_hash"] = settings.options_hash
            record_meta["version_hash"] = settings.version_hash

        return PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status="skipped",
            started_at=now,
            ended_at=now,
            duration_ms=0.0,
            attempts=0,
            error=reason,
            meta=record_meta,
        )


__all__ = [
    "BasePluginExecutor",
    "DefaultPluginExecutionStrategy",
    "ExecutionOptions",
    "ExecutionReportContext",
    "ExecutionStrategy",
    "ExecutionStrategyContext",
    "PluginExecutionStrategy",
]
