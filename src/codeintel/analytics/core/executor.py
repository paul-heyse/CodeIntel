"""Plugin executor for the unified plugin protocol.

This module provides the execution engine for plugins implementing
AnalyticsPluginProtocol. It handles:
- Plugin execution with error handling
- Retry logic with configurable policies
- Telemetry and contract validation
- Middleware chain for cross-cutting concerns
- Integration with the slim execution context
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from codeintel.analytics.core.contracts import (
    ContractValidationResult,
    PluginOutputContract,
    build_plugin_output_contracts,
    validate_plugin_outputs,
)
from codeintel.analytics.core.execution_context import (
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.analytics.core.plugin_protocol import (
    AnalyticsPluginProtocol,
    PluginExecutionRecord,
    PluginResult,
)
from codeintel.analytics.core.plugins.middleware.protocol import MiddlewareChain
from codeintel.analytics.core.registry import PluginPlan, PluginRegistry, get_registry
from codeintel.analytics.core.traits import is_contract_validated

if TYPE_CHECKING:
    from codeintel.analytics.core.plugins.middleware.protocol import PluginMiddleware

ExecutionStatus = Literal["succeeded", "failed", "partial"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Policy controlling plugin execution behavior.

    Attributes
    ----------
    fail_fast
        Stop execution on first failure.
    max_retries
        Maximum retry attempts for failed plugins.
    retry_backoff_ms
        Milliseconds to wait between retries.
    skip_on_unchanged
        Skip plugins whose inputs haven't changed.
    dry_run
        Plan but don't execute.
    validate_contracts
        Whether to validate output contracts.
    """

    fail_fast: bool = True
    max_retries: int = 0
    retry_backoff_ms: int = 100
    skip_on_unchanged: bool = False
    dry_run: bool = False
    validate_contracts: bool = True


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


class PluginExecutor:
    """Execute plugins with error handling, retries, and telemetry.

    The executor handles running plugins in dependency order, managing
    retries for transient failures, validating output contracts, and
    applying middleware for cross-cutting concerns.
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        *,
        policy: ExecutionPolicy | None = None,
        middleware: Sequence[PluginMiddleware] = (),
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
        """
        self._registry = registry or get_registry()
        self._policy = policy or ExecutionPolicy()
        self._middleware = MiddlewareChain(list(middleware))
        self._contract_cache: dict[str, tuple[PluginOutputContract, ...]] = {}

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
        run_id = ctx.run_id
        started_at = datetime.now(tz=UTC)
        records: list[PluginExecutionRecord] = []
        contract_results: dict[str, ContractValidationResult] = {}
        shared_scratch = scratch or PluginScratch()
        overall_status: ExecutionStatus = "succeeded"

        for plugin in plan.plugins:
            # Update context with plugin name and scratch
            plugin_ctx = self._prepare_context(ctx, plugin, shared_scratch)

            # Execute the plugin
            record = self._execute_plugin(plugin, plugin_ctx)
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
        return self._execute_plugin(plugin, ctx)

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
    ) -> PluginExecutionRecord:
        """Execute a single plugin with error handling and middleware.

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
        meta = plugin.metadata
        started_at = datetime.now(tz=UTC)

        # Check dry run
        if self._policy.dry_run:
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
        elif meta.severity == "skip_on_error":
            status = "skipped"
        else:
            status = "failed"

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
        max_attempts = max(self._policy.max_retries + 1, 1)
        attempts = 0
        error: str | None = None
        result: PluginResult | None = None
        start = time.perf_counter()

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
        """
        Return cached contracts for plugin or build them once.

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
        """
        Determine whether to run contract validation for a plugin.

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


def execute_plugin_plan(
    ctx: PluginExecutionContext,
    plan: PluginPlan,
    *,
    policy: ExecutionPolicy | None = None,
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
    "ExecutionPolicy",
    "ExecutionReport",
    "PluginExecutor",
    "execute_plugin_plan",
]
