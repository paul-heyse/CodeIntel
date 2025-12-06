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
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar, cast

from codeintel.core.execution.errors import PLUGIN_CATCHABLE_ERRORS
from codeintel.core.execution.telemetry import get_runtime_telemetry
from codeintel.core.execution.timing import utc_now
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.types.protocol import ValidationResult
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginResult

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
    ...     def _build_report(self, run_id, records, started_at, ended_at, ...):
    ...         return MyReport(...)
    """

    def __init__(
        self,
        policy: BaseExecutionPolicy | None = None,
        telemetry: RuntimeTelemetry | None = None,
        prior_manifest: Mapping[str, Mapping[str, object]] | None = None,
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
        """
        self._policy = policy or BaseExecutionPolicy()
        self._telemetry = telemetry or get_runtime_telemetry()
        self._prior_manifest = prior_manifest
        self._manifest: dict[str, dict[str, object]] = {}

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

    def _build_manifest_entry(
        self,
        plugin: P,
        record: PluginExecutionRecord,
        settings: PluginExecutionSettings | None,
    ) -> dict[str, object] | None:
        """Build manifest entry for a successful plugin execution.

        Override in subclass to enable manifest tracking. Base implementation
        returns None, which disables manifest tracking.

        Parameters
        ----------
        plugin
            Executed plugin.
        record
            Execution record.
        settings
            Plugin execution settings (contains hashes).

        Returns
        -------
        dict[str, object] | None
            Manifest entry if tracking enabled, None otherwise.
        """
        # Base implementation does nothing; subclasses override to enable
        _ = (plugin, record, settings)
        return None

    def _get_plugin_settings(
        self,
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
                settings = self._get_plugin_settings(plugin, settings_by_plugin)

                # Check for skip condition
                skip_reason = self._should_skip_plugin(plugin, executor_ctx)
                if skip_reason is not None:
                    record = self._create_skip_record(plugin, skip_reason)
                    records.append(record)
                    continue

                # Build plugin-specific context
                plugin_ctx = self._build_plugin_context(executor_ctx, plugin, shared_scratch)

                # Execute the plugin
                record = self._execute_single_plugin(
                    plugin,
                    plugin_ctx,
                    effective_run_id,
                    settings=settings,
                )
                records.append(record)

                # Build manifest entry for successful plugins
                if record.status == "succeeded":
                    manifest_entry = self._build_manifest_entry(plugin, record, settings)
                    if manifest_entry is not None:
                        self._manifest[plugin.metadata.name] = manifest_entry

                # Check for fail-fast condition
                if record.status == "failed":
                    # Use settings severity if available, otherwise plugin metadata
                    severity = (
                        settings.severity
                        if settings is not None
                        else self._policy.get_severity(plugin.metadata.name)
                    )
                    if severity == "fatal" and self._policy.fail_fast:
                        log.error(
                            "executor.plan.fatal_error plugin=%s",
                            plugin.metadata.name,
                        )
                        fatal_error = True
                        break
        finally:
            shared_scratch.cleanup()

        ended_at = utc_now()
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
        *,
        settings: PluginExecutionSettings | None = None,
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
        settings
            Optional per-plugin execution settings.

        Returns
        -------
        PluginExecutionRecord
            Execution record with status, duration, and result.
        """
        meta = plugin.metadata
        started_at = utc_now()
        start_time = time.perf_counter()

        # Build metadata for the record
        record_meta: dict[str, object] = {}
        if settings is not None:
            record_meta["input_hash"] = settings.input_hash
            record_meta["options_hash"] = settings.options_hash
            record_meta["version_hash"] = settings.version_hash

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
            ended_at = utc_now()
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._telemetry.end_span(span, success=False, error=validation_error)
            return PluginExecutionRecord(
                plugin_name=meta.name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                error=validation_error,
                meta=record_meta,
            )

        # Execute with retry
        result, attempts, error = self._execute_with_retries(plugin, ctx, settings)

        ended_at = utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Determine status - use settings severity if available
        severity = (
            settings.severity if settings is not None else self._policy.get_severity(meta.name)
        )

        if result is not None and result.success:
            status = "skipped" if result.skipped else "succeeded"
            self._on_plugin_success(plugin, ctx, result)
            rows_written = sum(result.row_counts.values()) if result.row_counts else 0
            self._telemetry.end_span(span, success=True, rows_written=rows_written)
        else:
            status = "skipped" if severity == "skip_on_error" else "failed"
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
            meta=record_meta,
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
        now = utc_now()
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
