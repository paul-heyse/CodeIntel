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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from codeintel.core.plugins.context import PluginScratch
from codeintel.core.plugins.policy import BaseExecutionPolicy
from codeintel.core.plugins.protocol import ValidationResult
from codeintel.core.plugins.result import PluginExecutionRecord, PluginResult
from codeintel.core.plugins.traits import get_retry_policy
from codeintel.core.runtime.errors import PLUGIN_CATCHABLE_ERRORS
from codeintel.core.runtime.telemetry import get_runtime_telemetry

if TYPE_CHECKING:
    from codeintel.core.plugins.context import PluginExecutionContext
    from codeintel.core.plugins.executor_context import BaseExecutorContext
    from codeintel.core.plugins.protocol import PluginProtocol
    from codeintel.core.plugins.registry import PluginPlan
    from codeintel.core.plugins.report import BaseExecutionReport
    from codeintel.core.runtime.telemetry import RuntimeTelemetry

log = logging.getLogger(__name__)

# Type variables for generic executor
P = TypeVar("P", bound="PluginProtocol")
C = TypeVar("C", bound="PluginExecutionContext")
EC = TypeVar("EC", bound="BaseExecutorContext")
R = TypeVar("R", bound="BaseExecutionReport")


class BasePluginExecutor(ABC, Generic[P, C, EC, R]):
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
    ...     def _build_report(self, run_id, records, started_at, ended_at, ...):
    ...         return MyReport(...)
    """

    def __init__(
        self,
        policy: BaseExecutionPolicy | None = None,
        telemetry: RuntimeTelemetry | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        policy
            Execution policy. Uses defaults if not provided.
        telemetry
            Runtime telemetry. Uses default singleton if not provided.
        """
        self._policy = policy or BaseExecutionPolicy()
        self._telemetry = telemetry or get_runtime_telemetry()

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
        run_id: str,
        records: list[PluginExecutionRecord],
        started_at: datetime,
        ended_at: datetime,
        duration_ms: float,
        fatal_error: bool,
        executor_ctx: EC,
    ) -> R:
        """Build domain-specific execution report.

        Parameters
        ----------
        run_id
            Unique identifier for this run.
        records
            Plugin execution records.
        started_at
            When execution started.
        ended_at
            When execution ended.
        duration_ms
            Total duration in milliseconds.
        fatal_error
            Whether execution ended due to fatal error.
        executor_ctx
            Executor context for additional report fields.

        Returns
        -------
        R
            Domain-specific execution report.
        """
        ...

    def _should_skip_plugin(
        self,
        plugin: P,
        executor_ctx: EC,
    ) -> str | None:
        """Check if plugin should be skipped.

        Override in subclass to implement domain-specific skip logic.

        Parameters
        ----------
        plugin
            Plugin to check.
        executor_ctx
            Executor context.

        Returns
        -------
        str | None
            Skip reason if plugin should be skipped, None otherwise.
        """
        # Base implementation ignores plugin and executor_ctx; subclasses use them
        _ = (plugin, executor_ctx)
        if self._policy.dry_run:
            return "dry_run"
        return None

    def _validate_plugin_inputs(self, plugin: P, ctx: C) -> tuple[bool, str | None]:
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
        tuple[bool, str | None]
            Tuple of (is_valid, error_message).
        """
        # Base uses self._policy (even though trivially) to ensure it's an instance method
        _ = self._policy  # Instance method validation hook
        # Check if plugin has validate_inputs method
        validate_method = getattr(plugin, "validate_inputs", None)
        if validate_method is not None and callable(validate_method):
            validation = cast("ValidationResult", validate_method(ctx))
            if not validation.valid:
                return False, f"Validation failed: {', '.join(validation.errors)}"
        return True, None

    def _on_plugin_success(self, plugin: P, ctx: C, result: PluginResult) -> None:
        """Process successful plugin execution.

        Override in subclass for post-execution processing.

        Parameters
        ----------
        plugin
            Executed plugin.
        ctx
            Plugin execution context.
        result
            Plugin result.
        """
        # Base hook does nothing; subclasses override
        _ = (plugin, ctx, result)

    def _on_plugin_failure(self, plugin: P, ctx: C, error: str) -> None:
        """Handle plugin execution failure.

        Override in subclass for failure handling.

        Parameters
        ----------
        plugin
            Failed plugin.
        ctx
            Plugin execution context.
        error
            Error message.
        """
        # Base hook does nothing; subclasses override
        _ = (plugin, ctx, error)

    def execute_plan(
        self,
        executor_ctx: EC,
        plan: PluginPlan[P],
        *,
        scratch: PluginScratch | None = None,
        run_id: str | None = None,
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

        Returns
        -------
        R
            Execution report with all plugin results.
        """
        effective_run_id = run_id or executor_ctx.effective_run_id or plan.plan_id
        started_at = datetime.now(tz=UTC)
        start_time = time.perf_counter()
        records: list[PluginExecutionRecord] = []
        shared_scratch = scratch or PluginScratch()
        fatal_error = False

        log.info(
            "executor.plan.start run_id=%s plugin_count=%d",
            effective_run_id,
            len(plan.plugins),
        )

        try:
            for plugin in plan.plugins:
                # Check for skip condition
                skip_reason = self._should_skip_plugin(plugin, executor_ctx)
                if skip_reason is not None:
                    record = self._create_skip_record(plugin, skip_reason)
                    records.append(record)
                    continue

                # Build plugin-specific context
                plugin_ctx = self._build_plugin_context(executor_ctx, plugin, shared_scratch)

                # Execute the plugin
                record = self._execute_single_plugin(plugin, plugin_ctx, effective_run_id)
                records.append(record)

                # Check for fail-fast condition
                if record.status == "failed":
                    severity = plugin.metadata.severity
                    if severity == "fatal" and self._policy.fail_fast:
                        log.error(
                            "executor.plan.fatal_error plugin=%s",
                            plugin.metadata.name,
                        )
                        fatal_error = True
                        break
        finally:
            shared_scratch.cleanup()

        ended_at = datetime.now(tz=UTC)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Record run-level telemetry
        self._telemetry.record_run_metrics(
            run_id=effective_run_id,
            success_count=sum(1 for r in records if r.status == "succeeded"),
            failure_count=sum(1 for r in records if r.status == "failed"),
            skip_count=sum(1 for r in records if r.status == "skipped"),
            duration_s=duration_ms / 1000,
        )

        log.info(
            "executor.plan.complete run_id=%s duration_ms=%.2f fatal_error=%s",
            effective_run_id,
            duration_ms,
            fatal_error,
        )

        return self._build_report(
            run_id=effective_run_id,
            records=records,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            fatal_error=fatal_error,
            executor_ctx=executor_ctx,
        )

    def _execute_single_plugin(
        self,
        plugin: P,
        ctx: C,
        run_id: str,
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

        Returns
        -------
        PluginExecutionRecord
            Execution record with status, duration, and result.
        """
        meta = plugin.metadata
        started_at = datetime.now(tz=UTC)
        start_time = time.perf_counter()

        # Start telemetry span
        span = self._telemetry.start_span(
            meta.name,
            run_id,
            attributes={"stage": meta.stage, "kind": meta.kind},
        )

        log.info(
            "executor.plugin.start name=%s stage=%s",
            meta.name,
            meta.stage,
        )

        # Validate inputs
        is_valid, validation_error = self._validate_plugin_inputs(plugin, ctx)
        if not is_valid:
            ended_at = datetime.now(tz=UTC)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._telemetry.end_span(span, success=False, error=validation_error)
            return PluginExecutionRecord(
                plugin_name=meta.name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                error=validation_error,
            )

        # Execute with retry
        result, attempts, error = self._execute_with_retries(plugin, ctx)

        ended_at = datetime.now(tz=UTC)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Determine status
        if result is not None and result.success:
            status = "skipped" if result.skipped else "succeeded"
            self._on_plugin_success(plugin, ctx, result)
            rows_written = sum(result.row_counts.values()) if result.row_counts else 0
            self._telemetry.end_span(span, success=True, rows_written=rows_written)
        else:
            status = "skipped" if meta.severity == "skip_on_error" else "failed"
            if error:
                self._on_plugin_failure(plugin, ctx, error)
            self._telemetry.end_span(span, success=False, error=error)

        log.info(
            "executor.plugin.complete name=%s status=%s duration_ms=%.2f attempts=%d",
            meta.name,
            status,
            duration_ms,
            attempts,
        )

        return PluginExecutionRecord(
            plugin_name=meta.name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            attempts=attempts,
            result=result,
            error=error,
        )

    def _execute_with_retries(
        self,
        plugin: P,
        ctx: C,
    ) -> tuple[PluginResult | None, int, str | None]:
        """Execute plugin with retry logic.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Plugin execution context.

        Returns
        -------
        tuple[PluginResult | None, int, str | None]
            Result, attempt count, and error message.
        """
        # Get retry policy for this plugin (may be custom or default)
        retry_policy = get_retry_policy(plugin)

        # If policy has retries configured, use that
        if self._policy.max_retries > 0:
            retry_policy = self._policy.to_retry_policy()

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
    def _create_skip_record(plugin: P, reason: str) -> PluginExecutionRecord:
        """Create a skip record for a plugin.

        Parameters
        ----------
        plugin
            Plugin that was skipped.
        reason
            Reason for skipping.

        Returns
        -------
        PluginExecutionRecord
            Record with skipped status.
        """
        now = datetime.now(tz=UTC)
        return PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status="skipped",
            started_at=now,
            ended_at=now,
            duration_ms=0.0,
            error=reason,
        )


__all__ = [
    "BasePluginExecutor",
]
