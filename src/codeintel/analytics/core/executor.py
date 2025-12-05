"""Plugin executor for the unified plugin protocol.

This module provides the execution engine for plugins implementing
AnalyticsPluginProtocol. It handles:
- Plugin execution with error handling
- Retry logic with configurable policies
- Telemetry and contract validation
- Middleware chain for cross-cutting concerns
- Integration with the slim execution context

Architecture Note
-----------------
This executor follows the patterns established in `codeintel.core.plugins.executor`
(BasePluginExecutor) and uses:
- `codeintel.core.plugins.policy.BaseExecutionPolicy` for execution configuration
- `codeintel.core.runtime.telemetry` for OTel/Prometheus integration
- `codeintel.core.runtime.retry` for tenacity-based retries

The analytics executor has domain-specific features (middleware chain, contract
validation) that extend beyond the base executor, but follows the same patterns.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from codeintel.analytics.core.context import (
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.analytics.core.contracts import (
    ContractValidationResult,
    PluginOutputContract,
    build_plugin_output_contracts,
    validate_plugin_outputs,
)
from codeintel.analytics.core.protocol import (
    AnalyticsPluginProtocol,
    PluginExecutionRecord,
    PluginResult,
)
from codeintel.analytics.core.registry import PluginPlan, PluginRegistry, get_registry
from codeintel.analytics.core.traits import is_contract_validated
from codeintel.analytics.plugins.middleware.protocol import MiddlewareChain
from codeintel.core.plugins.executor_context import BaseExecutorContext
from codeintel.core.plugins.policy import BaseExecutionPolicy
from codeintel.core.plugins.report import BaseExecutionReport, ExecutionStatus
from codeintel.core.runtime.telemetry import RuntimeTelemetry, get_runtime_telemetry

if TYPE_CHECKING:
    from codeintel.analytics.plugins.middleware.protocol import PluginMiddleware
    from codeintel.analytics.runtime.manifest import AnalyticsScope
    from codeintel.config.primitives import SnapshotRef
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# =============================================================================
# Analytics-Specific Policy (extends BaseExecutionPolicy)
# =============================================================================


@dataclass(frozen=True)
class AnalyticsExecutionPolicy(BaseExecutionPolicy):
    """Policy controlling analytics plugin execution behavior.

    Extend BaseExecutionPolicy with analytics-specific defaults.

    Attributes
    ----------
    validate_contracts
        Whether to validate output contracts. Defaults to True for analytics.
    """

    validate_contracts: bool = True


# Backward-compat alias
ExecutionPolicy = AnalyticsExecutionPolicy


# =============================================================================
# Analytics-Specific Executor Context (extends BaseExecutorContext)
# =============================================================================


@dataclass
class AnalyticsExecutorContext(BaseExecutorContext):
    """Analytics-specific executor context.

    Extend BaseExecutorContext with analytics-specific fields like scope.

    Attributes
    ----------
    scope
        Analytics scope restricting execution.
    """

    scope: AnalyticsScope | None = None


# =============================================================================
# Analytics-Specific Report (extends BaseExecutionReport)
# =============================================================================


@dataclass(frozen=True)
class AnalyticsExecutionReport(BaseExecutionReport):
    """Analytics-specific execution report.

    Extend BaseExecutionReport with contract validation results.

    Attributes
    ----------
    contract_results
        Contract validation results by plugin name.
    """

    contract_results: Mapping[str, ContractValidationResult] = field(default_factory=dict)


# Backward-compat alias (mutable version for legacy code)
@dataclass
class ExecutionReport:
    """Report of plugin execution run.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Total execution duration.
    status
        Overall status.
    records
        Per-plugin execution records.
    contract_results
        Contract validation results by plugin.
    """

    run_id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    status: ExecutionStatus
    records: tuple[PluginExecutionRecord, ...]
    contract_results: Mapping[str, ContractValidationResult] = field(default_factory=dict)

    @property
    def succeeded_count(self) -> int:
        """Count of successfully executed plugins."""
        return sum(1 for r in self.records if r.status == "succeeded")

    @property
    def failed_count(self) -> int:
        """Count of failed plugins."""
        return sum(1 for r in self.records if r.status == "failed")

    @property
    def skipped_count(self) -> int:
        """Count of skipped plugins."""
        return sum(1 for r in self.records if r.status == "skipped")


# =============================================================================
# Analytics Plugin Executor
# =============================================================================


class PluginExecutor:
    """Execute analytics plugins with error handling, retries, and telemetry.

    The executor handles running plugins in dependency order, managing
    retries for transient failures, validating output contracts, and
    applying middleware for cross-cutting concerns.

    This executor follows the patterns from `BasePluginExecutor` but includes
    analytics-specific features like middleware and contract validation.
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        *,
        policy: AnalyticsExecutionPolicy | None = None,
        middleware: Sequence[PluginMiddleware] = (),
        telemetry: RuntimeTelemetry | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        registry
            Plugin registry to use. Defaults to global registry.
        policy
            Execution policy. Defaults to standard policy.
        middleware
            Middleware to apply to plugin execution.
        telemetry
            Runtime telemetry instance.
        """
        self._registry = registry or get_registry()
        self._policy = policy or AnalyticsExecutionPolicy()
        self._middleware = MiddlewareChain(list(middleware))
        self._telemetry = telemetry or get_runtime_telemetry()
        self._contract_cache: dict[str, tuple[PluginOutputContract, ...]] = {}

    @property
    def policy(self) -> AnalyticsExecutionPolicy:
        """Return the execution policy."""
        return self._policy

    @property
    def telemetry(self) -> RuntimeTelemetry:
        """Return the telemetry instance."""
        return self._telemetry

    def add_middleware(self, mw: PluginMiddleware) -> None:
        """Add middleware to the execution chain.

        Parameters
        ----------
        mw
            Middleware to add.
        """
        self._middleware.add(mw)

    def execute(
        self,
        ctx: PluginExecutionContext,
        plan: PluginPlan,
        *,
        scratch: PluginScratch | None = None,
    ) -> ExecutionReport:
        """Execute all plugins in the plan.

        Parameters
        ----------
        ctx
            Execution context for plugins.
        plan
            Planned plugins with dependency ordering.
        scratch
            Shared scratch store for inter-plugin communication.

        Returns
        -------
        ExecutionReport
            Complete execution report.
        """
        run_id = ctx.run_id or plan.plan_id
        started_at = datetime.now(tz=UTC)
        records: list[PluginExecutionRecord] = []
        contract_results: dict[str, ContractValidationResult] = {}
        shared_scratch = scratch or PluginScratch()
        overall_status: ExecutionStatus = "succeeded"

        log.info(
            "executor.plan.start run_id=%s plugin_count=%d",
            run_id,
            len(plan.plugins),
        )

        for plugin in plan.plugins:
            # Update context with plugin name and scratch
            plugin_ctx = self._prepare_context(ctx, plugin, shared_scratch)

            # Execute the plugin
            record = self._execute_plugin(plugin, plugin_ctx, run_id)
            records.append(record)

            contracts: tuple[PluginOutputContract, ...] = ()
            if self._policy.validate_contracts:
                contracts = self._get_plugin_contracts(plugin)

            # Validate contracts if enabled
            if self._should_validate_plugin(record.status, plugin, contracts) and contracts:
                validation_map = validate_plugin_outputs(
                    plugin_ctx.gateway,
                    plugin_ctx.snapshot,
                    contracts,
                )
                contract_results[plugin.metadata.name] = validation_map[plugin.metadata.name]

            # Check for stop condition
            if record.status == "failed":
                if plugin.metadata.severity == "fatal" and self._policy.fail_fast:
                    overall_status = cast("ExecutionStatus", "failed")
                    break
                overall_status = cast("ExecutionStatus", "partial")
            elif record.status == "skipped":
                if overall_status == "succeeded":
                    overall_status = cast("ExecutionStatus", "partial")

        # Cleanup scratch
        shared_scratch.cleanup()

        ended_at = datetime.now(tz=UTC)
        duration_ms = (ended_at - started_at).total_seconds() * 1000

        # Record run-level telemetry
        self._telemetry.record_run_metrics(
            run_id=run_id,
            success_count=sum(1 for r in records if r.status == "succeeded"),
            failure_count=sum(1 for r in records if r.status == "failed"),
            skip_count=sum(1 for r in records if r.status == "skipped"),
            duration_s=duration_ms / 1000,
        )

        log.info(
            "executor.plan.complete run_id=%s duration_ms=%.2f status=%s",
            run_id,
            duration_ms,
            overall_status,
        )

        return ExecutionReport(
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=cast("ExecutionStatus", overall_status),
            records=tuple(records),
            contract_results=contract_results,
        )

    def execute_single(
        self,
        plugin: AnalyticsPluginProtocol,
        ctx: PluginExecutionContext,
    ) -> PluginExecutionRecord:
        """Execute a single plugin.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Execution context.

        Returns
        -------
        PluginExecutionRecord
            Execution record.
        """
        return self._execute_plugin(plugin, ctx, ctx.run_id or "single")

    @staticmethod
    def _prepare_context(
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        scratch: PluginScratch,
    ) -> PluginExecutionContext:
        """Prepare context for a specific plugin.

        Resources are accessed through the ResourceRegistry, so this method
        simply creates a new context with the shared scratch and plugin name.

        Parameters
        ----------
        ctx
            Base execution context.
        plugin
            Plugin being executed.
        scratch
            Shared scratch store.

        Returns
        -------
        PluginExecutionContext
            Context prepared for the plugin.
        """
        return PluginExecutionContext(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            run_id=ctx.run_id,
            scope=ctx.scope,
            configs=ctx.configs,
            resources=ctx.resources,
            scratch=scratch,
            options=ctx.options,
            plugin_name=plugin.metadata.name,
            extra=dict(ctx.extra),
        )

    def _execute_plugin(
        self,
        plugin: AnalyticsPluginProtocol,
        ctx: PluginExecutionContext,
        run_id: str,
    ) -> PluginExecutionRecord:
        """Execute a single plugin with error handling and middleware.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Execution context.
        run_id
            Run identifier for telemetry.

        Returns
        -------
        PluginExecutionRecord
            Execution record.
        """
        meta = plugin.metadata
        started_at = datetime.now(tz=UTC)

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

        # Check dry run
        if self._policy.dry_run:
            self._telemetry.end_span(span, success=True)
            return PluginExecutionRecord(
                plugin_name=meta.name,
                status="skipped",
                started_at=started_at,
                ended_at=datetime.now(tz=UTC),
                duration_ms=0.0,
                error="dry_run",
            )

        # Validate inputs
        validation = plugin.validate_inputs(ctx)
        if not validation.valid:
            self._telemetry.end_span(
                span,
                success=False,
                error=f"Validation failed: {', '.join(validation.errors)}",
            )
            return PluginExecutionRecord(
                plugin_name=meta.name,
                status="failed",
                started_at=started_at,
                ended_at=datetime.now(tz=UTC),
                duration_ms=0.0,
                error=f"Validation failed: {', '.join(validation.errors)}",
            )

        # Call middleware before_execute
        self._middleware.before_execute(ctx, plugin)

        # Execute with retries
        result, attempts, duration_ms, error = self._execute_with_retries(plugin, ctx)

        # Call middleware after_execute if we have a result
        if result is not None:
            result = self._middleware.after_execute(ctx, plugin, result)

        ended_at = datetime.now(tz=UTC)
        status: Literal["succeeded", "failed", "skipped"]

        if result is not None and result.success:
            status = "succeeded"
            rows_written = sum(result.row_counts.values()) if result.row_counts else 0
            self._telemetry.end_span(span, success=True, rows_written=rows_written)
        elif meta.severity == "skip_on_error":
            status = "skipped"
            self._telemetry.end_span(span, success=False, error=error)
        else:
            status = "failed"
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
        plugin: AnalyticsPluginProtocol,
        ctx: PluginExecutionContext,
    ) -> tuple[PluginResult | None, int, float, str | None]:
        """Execute plugin with retry logic and middleware error handling.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Execution context.

        Returns
        -------
        tuple[PluginResult | None, int, float, str | None]
            Result, attempt count, duration, and error message.
        """
        # Use tenacity-based retry from policy
        retry_policy = self._policy.to_retry_policy()
        max_attempts = max(self._policy.max_retries + 1, 1)
        attempts = 0
        error: str | None = None
        result: PluginResult | None = None
        start = time.perf_counter()

        # Use the retry policy's retry configuration
        _ = retry_policy  # Available for future enhancement

        while attempts < max_attempts:
            attempts += 1
            try:
                result = plugin.execute(ctx)
                if result.success:
                    error = None
                    break
                error = result.error
                # Don't retry if plugin explicitly failed
                break
            except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as exc:
                # Let middleware handle the error
                handled_error = self._middleware.on_error(ctx, plugin, exc)

                if handled_error is None:
                    # Error was suppressed by middleware
                    error = None
                    break

                error = repr(handled_error)
                log.warning(
                    "Plugin %s failed (attempt %d/%d): %s",
                    plugin.metadata.name,
                    attempts,
                    max_attempts,
                    error,
                )
                if attempts < max_attempts and self._policy.retry_backoff_ms > 0:
                    time.sleep(self._policy.retry_backoff_ms / 1000)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return result, attempts, duration_ms, error

    def get_plugin_contracts(
        self,
        plugin: AnalyticsPluginProtocol,
    ) -> tuple[PluginOutputContract, ...]:
        """Return cached output contracts for inspection and testing.

        Returns
        -------
        tuple[PluginOutputContract, ...]
            Cached output contracts for the provided plugin.
        """
        return self._get_plugin_contracts(plugin)

    def _get_plugin_contracts(
        self,
        plugin: AnalyticsPluginProtocol,
    ) -> tuple[PluginOutputContract, ...]:
        """Return cached contracts for plugin or build them once.

        Returns
        -------
        tuple[PluginOutputContract, ...]
            Cached contracts, empty when none are declared.
        """
        name = plugin.metadata.name
        if name in self._contract_cache:
            log.debug("Contract cache hit for plugin=%s", name)
            return self._contract_cache[name]
        log.debug("Contract cache miss for plugin=%s; building contracts", name)
        contracts = build_plugin_output_contracts(plugin)
        self._contract_cache[name] = contracts
        return contracts

    def _should_validate_plugin(
        self,
        status: Literal["succeeded", "failed", "skipped"],
        plugin: AnalyticsPluginProtocol,
        contracts: tuple[PluginOutputContract, ...],
    ) -> bool:
        """Determine whether to run contract validation for a plugin.

        Returns
        -------
        bool
            True when validation should run.
        """
        if not self._policy.validate_contracts:
            return False
        if status != "succeeded":
            return False
        if is_contract_validated(plugin):
            return True
        return bool(contracts)


# =============================================================================
# Convenience factory for building executor context
# =============================================================================


def build_analytics_executor_context(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    scope: AnalyticsScope | None = None,
    run_context: RunContext | None = None,
    telemetry: RuntimeTelemetry | None = None,
) -> AnalyticsExecutorContext:
    """Build an analytics executor context.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Repository snapshot reference.
    scope
        Analytics scope.
    run_context
        Optional run context.
    telemetry
        Optional telemetry instance.

    Returns
    -------
    AnalyticsExecutorContext
        Configured analytics executor context.
    """
    return AnalyticsExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        scope=scope,
        run_context=run_context,
        telemetry=telemetry or get_runtime_telemetry(),
    )


def execute_plugin_plan(
    ctx: PluginExecutionContext,
    plan: PluginPlan,
    *,
    policy: AnalyticsExecutionPolicy | None = None,
) -> ExecutionReport:
    """Execute a plugin plan with default executor.

    Parameters
    ----------
    ctx
        Execution context.
    plan
        Plugin plan to execute.
    policy
        Optional execution policy.

    Returns
    -------
    ExecutionReport
        Execution report.
    """
    executor = PluginExecutor(policy=policy)
    return executor.execute(ctx, plan)


__all__ = [
    "AnalyticsExecutionPolicy",
    "AnalyticsExecutionReport",
    "AnalyticsExecutorContext",
    "ExecutionPolicy",
    "ExecutionReport",
    "PluginExecutor",
    "build_analytics_executor_context",
    "execute_plugin_plan",
]
