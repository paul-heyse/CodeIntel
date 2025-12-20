"""Hamilton-based build executor.

This module provides HamiltonBuildExecutor, a DAG-based executor for build
targets using Hamilton's Driver.

Design Principles
-----------------
1. HamiltonBuildExecutor.run() is the main entry point for execution.
2. It maps target names to Hamilton node names via runtime mappings.
3. Results are returned in a structured HamiltonBuildResult.
4. The executor integrates with existing manifest/tracking infrastructure.
5. Executes the full dependency closure, not just requested targets.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import hamilton.base as h_base

from codeintel.build.execution_policy import ExecutionPolicy
from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
from codeintel.build.hamilton.contracts.enforced_gateway import ContractEnforcingStorageGateway
from codeintel.build.hamilton.driver_factory import build_driver, target_to_node_name
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.hooks import NodeTelemetryHook, build_hooks
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.run_writer import BuildRunWriter

if TYPE_CHECKING:
    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunState:
    """Execution state shared across run steps."""

    env: BuildEnv
    targets: tuple[str, ...]
    runtime: HamiltonRuntime
    run_id: str
    start_time: float
    started_at: datetime

    @property
    def duration_ms(self) -> float:
        """Return elapsed milliseconds for the run."""
        return (time.perf_counter() - self.start_time) * 1000


def _generate_run_id() -> str:
    """Generate a unique run ID for build tracking.

    Returns
    -------
    str
        Unique run identifier for this Hamilton execution.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"hamilton-{timestamp}-{suffix}"


def _categorize_outputs(
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    runtime: HamiltonRuntime,
) -> tuple[list[str], list[str], list[str]]:
    """Categorize outputs into computed/skipped/failed lists.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Computed, skipped, and failed targets in that order.
    """
    computed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for target in closure:
        node_name = target_to_node_name(target, runtime=runtime)
        if node_name is None:
            failed.append(target)
            continue

        record = outputs.get(node_name)
        if not isinstance(record, TargetRunRecord):
            failed.append(target)
        elif record.status == "succeeded":
            computed.append(target)
        elif record.status == "skipped":
            skipped.append(target)
        else:
            failed.append(target)

    return computed, skipped, failed


def _map_closure_to_nodes(
    closure: tuple[str, ...],
    runtime: HamiltonRuntime,
) -> tuple[list[str], list[str]]:
    """Map closure targets to Hamilton node names.

    Returns
    -------
    tuple[list[str], list[str]]
        Node names for final execution variables, and missing targets.
    """
    final_vars: list[str] = []
    missing: list[str] = []

    for target in closure:
        node_name = target_to_node_name(target, runtime=runtime)
        if node_name is None:
            missing.append(target)
        else:
            final_vars.append(node_name)

    return final_vars, missing


@dataclass(frozen=True)
class HamiltonBuildResult:
    """Result of a Hamilton-based build execution.

    Attributes
    ----------
    requested
        Tuple of target names that were requested by the user.
    closure
        Tuple of target names in the full dependency closure.
    computed_targets
        Targets that were actually computed (status="succeeded").
    skipped_targets
        Targets that were skipped (status="skipped").
    failed_targets
        Targets that failed during execution (status="failed").
    outputs
        Dictionary mapping Hamilton node names to their outputs.
    success
        Whether all requested targets succeeded.
    duration_ms
        Total execution duration in milliseconds.
    error
        Error message if the entire execution failed.
    run_id
        Unique identifier for this build run.
    runtime
        Reference to the HamiltonRuntime for mapping lookups.
    """

    requested: tuple[str, ...]
    closure: tuple[str, ...] = ()
    computed_targets: tuple[str, ...] = ()
    skipped_targets: tuple[str, ...] = ()
    failed_targets: tuple[str, ...] = ()
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration_ms: float = 0.0
    error: str | None = None
    run_id: str = ""
    runtime: HamiltonRuntime | None = None

    def get_record(self, target_name: str) -> TargetRunRecord | None:
        """Get the execution record for a target.

        Returns
        -------
        TargetRunRecord | None
            Execution record for the target, if present.
        """
        node_name = target_to_node_name(target_name, runtime=self.runtime)
        if node_name is None:
            return None
        value = self.outputs.get(node_name)
        return value if isinstance(value, TargetRunRecord) else None


class HamiltonBuildExecutor:
    """Execute build targets using Hamilton Driver.

    Parameters
    ----------
    profile
        Optional policy profile name (e.g., "fast", "full", "default").
    enable_cache
        When True, enable Hamilton's on-disk caching adapter for nodes decorated with
        ``@cache``.
    cache_dir
        Optional override for the Hamilton cache directory. When omitted, uses
        ``{build_dir}/.hamilton_cache`` from the provided BuildEnv.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        parallel_backend: str = "sequential",
        max_workers: int | None = None,
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the Hamilton executor."""
        self._options = BuildExecutionOptions(
            profile=profile,
            parallel_backend=parallel_backend,
            max_workers=max_workers,
            enable_hamilton_cache=enable_cache,
            cache_dir=cache_dir,
        )

    @property
    def profile(self) -> str | None:
        """Return the configured profile name."""
        return self._options.profile

    def run(
        self,
        *,
        env: BuildEnv,
        targets: list[str],
    ) -> HamiltonBuildResult:
        """Execute build targets using Hamilton.

        Returns
        -------
        HamiltonBuildResult
            Structured result containing outputs and status details.
        """
        run_id = _generate_run_id()
        writer = BuildRunWriter(env.gateway)
        runtime, telemetry_hook = self._build_runtime(env=env, run_id=run_id, writer=writer)

        context = _RunState(
            env=env,
            targets=tuple(targets),
            runtime=runtime,
            run_id=run_id,
            start_time=time.perf_counter(),
            started_at=datetime.now(tz=UTC),
        )
        return self._run_with_state(
            context=context,
            writer=writer,
            telemetry_hook=telemetry_hook,
        )

    def _run_with_state(
        self,
        *,
        context: _RunState,
        writer: BuildRunWriter,
        telemetry_hook: NodeTelemetryHook | None,
    ) -> HamiltonBuildResult:
        graph = context.runtime.graph
        requested_targets = list(context.targets)

        log.info(
            "build.hamilton.executor.start run_id=%s targets=%s",
            context.run_id,
            requested_targets,
        )

        writer.start_run(
            env=context.env,
            run_id=context.run_id,
            requested_targets=requested_targets,
            started_at=context.started_at,
        )

        closure = self._compute_closure(graph, requested_targets, context.run_id)
        if closure is None:
            writer.complete_run(
                run_id=context.run_id,
                success=False,
                computed_targets=(),
                skipped_targets=(),
                error_summary="Failed to compute closure",
            )
            return self._make_error_result(context, "Failed to compute closure")

        final_vars, missing = _map_closure_to_nodes(closure, context.runtime)
        if missing:
            writer.complete_run(
                run_id=context.run_id,
                success=False,
                computed_targets=(),
                skipped_targets=(),
                error_summary=f"Missing node mappings for: {missing}",
            )
            return self._make_missing_result(context, closure, missing)

        try:
            outputs, error = self._execute_dag(
                context.runtime,
                final_vars,
                context.env,
                context.run_id,
                graph=graph,
            )
        finally:
            if telemetry_hook is not None:
                telemetry_hook.flush()

        computed, skipped, failed = _categorize_outputs(closure, outputs, context.runtime)
        duration_ms = context.duration_ms
        success = not failed and error is None

        records: list[TargetRunRecord] = [
            value for value in outputs.values() if isinstance(value, TargetRunRecord)
        ]
        writer.save_run_targets(env=context.env, run_id=context.run_id, records=records)
        writer.persist_asset_catalog(
            env=context.env,
            run_id=context.run_id,
            graph=graph,
            records=records,
        )

        writer.complete_run(
            run_id=context.run_id,
            success=success,
            computed_targets=computed,
            skipped_targets=skipped,
            error_summary=error or (f"{len(failed)} targets failed" if failed else None),
        )

        log.info(
            "build.hamilton.executor.complete run_id=%s success=%s duration_ms=%.1f",
            context.run_id,
            success,
            duration_ms,
        )

        return HamiltonBuildResult(
            requested=context.targets,
            closure=closure,
            computed_targets=tuple(computed),
            skipped_targets=tuple(skipped),
            failed_targets=tuple(failed),
            outputs=outputs,
            success=success,
            duration_ms=duration_ms,
            error=error,
            run_id=context.run_id,
            runtime=context.runtime,
        )

    def _effective_max_workers(self, graph: TargetGraph) -> int | None:
        limits: list[int] = []
        for target in graph.all_targets:
            policy = ExecutionPolicy(run_options=self._options, target_execution=target.execution)
            max_workers = policy.effective_max_workers()
            if max_workers is not None:
                limits.append(max_workers)
        if not limits:
            return self._options.max_workers
        return min(limits)

    def _build_runtime(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        writer: BuildRunWriter,
    ) -> tuple[HamiltonRuntime, NodeTelemetryHook | None]:
        """Build Hamilton runtime with configured mode and lifecycle adapters.

        Returns
        -------
        HamiltonRuntime
            Configured runtime with driver and target graph.
        """
        config: dict[str, Any] = {"profile": self._options.resolved_profile(env=env)}
        telemetry_hook: NodeTelemetryHook | None = None

        hook_options = self._options.hook_options(env=env)

        def _adapter_factory(graph: TargetGraph) -> list[LifecycleAdapter]:
            nonlocal telemetry_hook
            adapters: list[LifecycleAdapter] = []
            effective_max_workers = self._effective_max_workers(graph)
            parallel_adapter = create_parallel_adapter(
                self._options.parallel_backend,
                max_workers=effective_max_workers,
                thread_name_prefix="codeintel-build",
            )
            if parallel_adapter is not None:
                adapters.append(parallel_adapter)
            else:
                adapters.append(h_base.DictResult())

            hooks = build_hooks(run_id, writer, graph, options=hook_options)
            for hook in hooks:
                if isinstance(hook, NodeTelemetryHook):
                    telemetry_hook = hook
                adapters.append(cast("LifecycleAdapter", hook))
            return adapters

        runtime = build_driver(
            config=config,
            adapter_factory=_adapter_factory,
            enable_cache=self._options.enable_hamilton_cache,
            cache_dir=str(self._options.resolved_cache_dir(env=env)),
        )
        return runtime, telemetry_hook

    @staticmethod
    def _compute_closure(
        graph: TargetGraph,
        targets: list[str],
        run_id: str,
    ) -> tuple[str, ...] | None:
        """Compute dependency closure, returning None on error.

        Returns
        -------
        tuple[str, ...] | None
            Ordered dependency closure, or None if computation failed.
        """
        try:
            return graph.topological_order(targets)
        except (KeyError, ValueError):
            log.exception("build.hamilton.executor.closure_error run_id=%s", run_id)
            return None

    @staticmethod
    def _make_error_result(
        context: _RunState,
        error: str,
    ) -> HamiltonBuildResult:
        """Create error result for closure computation failure.

        Returns
        -------
        HamiltonBuildResult
            Error result indicating failed closure computation.
        """
        return HamiltonBuildResult(
            requested=context.targets,
            success=False,
            failed_targets=context.targets,
            duration_ms=context.duration_ms,
            error=error,
            run_id=context.run_id,
            runtime=context.runtime,
        )

    @staticmethod
    def _make_missing_result(
        context: _RunState,
        closure: tuple[str, ...],
        missing: list[str],
    ) -> HamiltonBuildResult:
        """Create error result for missing node mappings.

        Returns
        -------
        HamiltonBuildResult
            Error result indicating missing node mappings.
        """
        log.error("build.hamilton.executor.missing_targets targets=%s", missing)
        return HamiltonBuildResult(
            requested=context.targets,
            closure=closure,
            success=False,
            failed_targets=tuple(missing),
            duration_ms=context.duration_ms,
            error=f"Missing node mappings for: {missing}",
            run_id=context.run_id,
            runtime=context.runtime,
        )

    @staticmethod
    def _execute_dag(
        runtime: HamiltonRuntime,
        final_vars: list[str],
        env: BuildEnv,
        run_id: str,
        *,
        graph: TargetGraph,
    ) -> tuple[dict[str, Any], str | None]:
        """Execute the Hamilton DAG, returning (outputs, error).

        Parameters
        ----------
        runtime
            Hamilton runtime with driver and graph.
        final_vars
            List of node names to execute.
        env
            Build environment.
        run_id
            Run identifier for tracking.
        graph
            Target graph for dependency and contract lookups.

        Returns
        -------
        tuple[dict[str, Any], str | None]
            Outputs keyed by node name, and optional error string.
        """
        try:
            execution_env = env
            if env.strict_contracts:
                wrapped_gateway = ContractEnforcingStorageGateway(env.gateway)
                execution_env = replace(
                    env,
                    gateway=cast("StorageGateway", wrapped_gateway),
                )

            outputs = runtime.dr.execute(
                list(final_vars),
                inputs={"env": execution_env, "graph": graph},
            )
        except Exception as exc:
            log.exception("build.hamilton.executor.error run_id=%s", run_id)
            return {}, str(exc)
        else:
            return outputs, None


__all__ = [
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
]
