"""Hamilton-based build executor.

This module provides HamiltonBuildExecutor, which is a drop-in alternative
to the legacy BuildExecutor. It uses Hamilton's Driver for DAG-based
execution of build targets.

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

from hamilton import base as h_base

from codeintel.build.assets.emitter import persist_asset_catalog_for_run
from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
from codeintel.build.hamilton.contracts.enforced_gateway import ContractEnforcingStorageGateway
from codeintel.build.hamilton.contracts.enforcement_hook import ContractEnforcementHook
from codeintel.build.hamilton.driver_factory import build_driver, target_to_node_name
from codeintel.build.hamilton.introspect import target_graph_from_hamilton
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.telemetry_hook import NodeTelemetryHook
from codeintel.build.manifest import BuildRunRecord
from codeintel.build.targets import TargetGraph
from codeintel.build.unified_registry import get_unified_registry
from codeintel.storage.exceptions import StorageError

if TYPE_CHECKING:
    from hamilton.lifecycle.base import LifecycleAdapter

    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode, HamiltonRuntime
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunContext:
    """Execution context shared across run steps."""

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


def _start_build_run(
    env: BuildEnv,
    run_id: str,
    targets: list[str],
    start_datetime: datetime,
) -> None:
    """Record the start of a build run."""
    try:
        record = BuildRunRecord(
            run_id=run_id,
            repo=env.repo,
            commit=env.commit,
            requested_targets=tuple(targets),
            computed_targets=(),
            skipped_targets=(),
            started_at=start_datetime,
            status="running",
        )
        env.gateway.build.start_run(record)
    except StorageError as exc:
        log.warning("build.hamilton.executor.start_run_failed run_id=%s error=%s", run_id, exc)


@dataclass(frozen=True)
class _RunCompletionParams:
    """Parameters for completing a build run."""

    env: BuildEnv
    run_id: str
    success: bool
    computed: tuple[str, ...]
    skipped: tuple[str, ...]
    error_summary: str | None


def _complete_build_run(params: _RunCompletionParams) -> None:
    """Complete a build run record."""
    try:
        status = "succeeded" if params.success else "failed"
        params.env.gateway.build.complete_run(
            run_id=params.run_id,
            status=status,
            computed_targets=params.computed,
            skipped_targets=params.skipped,
            error_summary=params.error_summary,
        )
    except StorageError as exc:
        log.warning(
            "build.hamilton.executor.complete_run_failed run_id=%s error=%s", params.run_id, exc
        )


def _persist_run_targets(
    env: BuildEnv,
    run_id: str,
    outputs: dict[str, Any],
) -> None:
    """Persist per-target execution records.

    Parameters
    ----------
    env
        Build environment with gateway.
    run_id
        Run identifier.
    outputs
        Outputs from Hamilton execution.
    """
    try:
        records: list[TargetRunRecord] = [
            value for value in outputs.values() if isinstance(value, TargetRunRecord)
        ]

        records.sort(key=lambda record: record.target)

        if records:
            env.gateway.build.save_run_targets(
                run_id=run_id,
                repo=env.repo,
                commit=env.commit,
                records=records,
            )
            log.debug(
                "build.hamilton.executor.run_targets_saved run_id=%s count=%d",
                run_id,
                len(records),
            )
    except StorageError as exc:
        log.warning("build.hamilton.executor.run_targets_failed run_id=%s error=%s", run_id, exc)


def _persist_asset_catalog(
    env: BuildEnv,
    run_id: str,
    outputs: dict[str, Any],
    graph: TargetGraph,
) -> None:
    """Persist Phase 4 asset catalog records (versions + lineage + run mapping)."""
    try:
        records: list[TargetRunRecord] = [
            value for value in outputs.values() if isinstance(value, TargetRunRecord)
        ]
        if not records:
            return
        records.sort(key=lambda record: record.target)

        persist_asset_catalog_for_run(
            env=env,
            run_id=run_id,
            graph=graph,
            records=records,
        )
    except StorageError as exc:
        log.warning("build.hamilton.executor.asset_catalog_failed run_id=%s error=%s", run_id, exc)


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
    mode
        Node mode: "phase0" for explicit nodes, "generated" for all targets.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        mode: HamiltonNodeMode = "generated",
        parallel_backend: str = "sequential",
        max_workers: int | None = None,
    ) -> None:
        """Initialize the Hamilton executor."""
        self._profile = profile
        self._mode: HamiltonNodeMode = mode
        self._parallel_backend = parallel_backend
        self._max_workers = max_workers

    @property
    def profile(self) -> str | None:
        """Return the configured profile name."""
        return self._profile

    @property
    def mode(self) -> HamiltonNodeMode:
        """Return the configured node mode."""
        return self._mode

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
        telemetry_hook = NodeTelemetryHook(run_id, env.gateway)
        runtime = self._build_runtime(
            env=env,
            telemetry_hook=telemetry_hook,
        )
        graph = target_graph_from_hamilton(runtime)

        context = _RunContext(
            env=env,
            targets=tuple(targets),
            runtime=runtime,
            run_id=run_id,
            start_time=time.perf_counter(),
            started_at=datetime.now(tz=UTC),
        )
        log.info(
            "build.hamilton.executor.start run_id=%s targets=%s mode=%s",
            context.run_id,
            targets,
            self._mode,
        )

        closure = self._compute_closure(graph, targets, context.run_id)
        if closure is None:
            return self._make_error_result(context, "Failed to compute closure")

        final_vars, missing = _map_closure_to_nodes(closure, context.runtime)
        if missing:
            return self._make_missing_result(context, closure, missing)

        _start_build_run(context.env, context.run_id, targets, context.started_at)

        outputs, error = self._execute_dag(
            context.runtime,
            final_vars,
            context.env,
            context.run_id,
            graph=graph,
        )

        # Flush telemetry after execution
        if telemetry_hook:
            telemetry_hook.flush()

        computed, skipped, failed = _categorize_outputs(closure, outputs, context.runtime)
        duration_ms = context.duration_ms
        success = not failed and error is None

        _persist_run_targets(context.env, context.run_id, outputs)
        _persist_asset_catalog(context.env, context.run_id, outputs, graph)

        _complete_build_run(
            _RunCompletionParams(
                env=context.env,
                run_id=context.run_id,
                success=success,
                computed=tuple(computed),
                skipped=tuple(skipped),
                error_summary=error or (f"{len(failed)} targets failed" if failed else None),
            )
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

    def _build_runtime(
        self,
        *,
        env: BuildEnv,
        telemetry_hook: NodeTelemetryHook,
    ) -> HamiltonRuntime:
        """Build Hamilton runtime with configured mode and lifecycle adapters.

        Returns
        -------
        HamiltonRuntime
            Configured runtime with driver and target graph.
        """
        config: dict[str, Any] = {"profile": self._profile or "default"}
        adapters = self._build_adapters(env=env, telemetry_hook=telemetry_hook)
        return build_driver(config=config, mode=self._mode, adapter=adapters)

    @staticmethod
    def _build_contract_graph() -> TargetGraph:
        unified = get_unified_registry()
        graph = TargetGraph()
        for target in unified.get_all_targets():
            graph.register(target)
        return graph

    def _build_adapters(
        self,
        *,
        env: BuildEnv,
        telemetry_hook: NodeTelemetryHook,
    ) -> list[LifecycleAdapter]:
        adapters: list[LifecycleAdapter] = []

        parallel_adapter = create_parallel_adapter(
            self._parallel_backend,
            max_workers=self._max_workers,
            thread_name_prefix="codeintel-build",
        )

        if parallel_adapter is not None:
            adapters.append(parallel_adapter)
        else:
            adapters.append(h_base.DictResult())

        if env.strict_contracts:
            adapters.append(ContractEnforcementHook(self._build_contract_graph(), strict=True))

        adapters.append(telemetry_hook)
        return adapters

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
        context: _RunContext,
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
        context: _RunContext,
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
