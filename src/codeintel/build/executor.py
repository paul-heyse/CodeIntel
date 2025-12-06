"""Build plan execution for the CodeIntel build system.

This module bridges the build system planning layer (Phases 1-4) to actual
execution via the domain modules (ingestion, graphs, analytics). The BuildExecutor
translates BuildPlan into module-specific calls, records manifests for
completed targets, and handles partial failures gracefully.

Key Components
--------------
- **BuildResult**: Final result of executing a build plan
- **StageExecutionResult**: Internal result from executing one stage
- **BuildExecutor**: Orchestrates execution of build plans

Integration Points
------------------
- Analytics: Direct plugin execution via `_execute_analytics_stage()`
- Graphs: `plan_graph_plugin_run()` + `GraphPluginExecutor`
- Ingestion: Direct plugin execution via `_execute_target_direct()`
- Tracking: `BuildTracking` for manifest and run record persistence
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.analytics.plugins.registration import ALL_PLUGINS
from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.errors import BuildErrorCollection, PluginExecutionError
from codeintel.build.hashing import compute_input_hash
from codeintel.build.manifest import BuildRunRecord, BuildStatus, OutputManifest
from codeintel.build.plan import BuildPlan, PlanStage
from codeintel.build.plugin_registry import get_plugin_for_target
from codeintel.build.providers import Providers, create_default_providers
from codeintel.build.targets import TargetGraph, TargetModule
from codeintel.config.steps_graphs import GraphPluginPolicy
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.export.export_jsonl import ExportCallOptions, export_all_jsonl
from codeintel.export.export_parquet import export_all_parquet
from codeintel.graphs.runtime.graph_executor import GraphExecutorContext, GraphPluginExecutor
from codeintel.graphs.runtime.planning import GraphPlanContext, plan_graph_plugin_run

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


@dataclass(frozen=True)
class StageExecutionResult:
    """Result from executing a single build stage.

    Internal type used by BuildExecutor to track per-stage outcomes
    before aggregating into the final BuildResult.

    Attributes
    ----------
    module
        Target module that was executed.
    completed
        Targets that completed successfully.
    failed
        Targets that failed during execution.
    durations_ms
        Execution duration per target in milliseconds.
    row_counts
        Row counts per target (None if unavailable).
    error
        Error message if stage failed, None otherwise.

    Examples
    --------
    >>> result = StageExecutionResult(
    ...     module="analytics",
    ...     completed=("function_metrics", "hotspots"),
    ...     failed=(),
    ...     durations_ms={"function_metrics": 5000, "hotspots": 2000},
    ...     row_counts={"function_metrics": 1500, "hotspots": 50},
    ...     error=None,
    ... )
    >>> len(result.completed)
    2
    """

    module: TargetModule
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    durations_ms: dict[str, float] = field(default_factory=dict)
    row_counts: dict[str, int | None] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True if no targets failed and no error occurred.

        Returns
        -------
        bool
            True if stage succeeded.
        """
        return len(self.failed) == 0 and self.error is None


@dataclass(frozen=True)
class BuildResult:
    """Result of executing a build plan.

    Public result type returned by BuildExecutor.execute(), containing
    full information about the build run including which targets
    completed, failed, or were skipped.

    Supports continue-and-collect semantics: errors are collected during
    execution and reported at the end, rather than failing on the first error.

    Attributes
    ----------
    run_id
        Unique identifier for this build run.
    plan
        The build plan that was executed.
    status
        Final status: "succeeded" or "failed".
    completed_targets
        Targets that were successfully computed.
    failed_targets
        Targets that failed during execution.
    skipped_targets
        Targets that were skipped (already current).
    duration_ms
        Total execution duration in milliseconds.
    error_summary
        Summary of errors if failed, None otherwise.
    errors
        Collection of all errors encountered during execution.

    Examples
    --------
    >>> result = executor.execute(plan)
    >>> if result.success:
    ...     print(f"Built {len(result.completed_targets)} targets")
    ... else:
    ...     print(f"Failed: {result.error_summary}")
    ...     for error in result.errors.errors:
    ...         print(f"  - {error.user_message}")
    """

    run_id: str
    plan: BuildPlan
    status: BuildStatus
    completed_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]
    skipped_targets: tuple[str, ...]
    duration_ms: float
    error_summary: str | None = None
    errors: BuildErrorCollection = field(default_factory=BuildErrorCollection)

    @property
    def success(self) -> bool:
        """Return True if the build succeeded.

        Returns
        -------
        bool
            True if status is "succeeded".
        """
        return self.status == "succeeded"

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were collected.

        Returns
        -------
        bool
            True if the error collection has errors.
        """
        return self.errors.has_errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary for JSON output.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "run_id": self.run_id,
            "status": self.status,
            "completed_targets": list(self.completed_targets),
            "failed_targets": list(self.failed_targets),
            "skipped_targets": list(self.skipped_targets),
            "duration_ms": self.duration_ms,
            "error_summary": self.error_summary,
            "error_count": len(self.errors),
            "plan": self.plan.to_dict(),
        }


# =============================================================================
# Build Executor
# =============================================================================


class BuildExecutor:
    """Execute build plans via the domain module system.

    Translates a BuildPlan into module-specific execution calls,
    orchestrating ingestion, graphs, and analytics stages. Records
    manifests for completed targets and handles partial failures.

    Parameters
    ----------
    graph
        Target graph for looking up target metadata.
    gateway
        Storage gateway for database access and manifest tracking.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    paths
        Build paths configuration.
    tools
        Tools configuration for build execution.

    Examples
    --------
    >>> executor = BuildExecutor(
    ...     graph=get_target_graph(),
    ...     gateway=gateway,
    ...     snapshot=snapshot,
    ...     paths=paths,
    ...     tools=tools,
    ... )
    >>> result = executor.execute(plan)
    >>> print(f"Completed: {result.completed_targets}")
    """

    def __init__(  # noqa: PLR0913 - Core dependencies for build execution
        self,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        paths: BuildPaths,
        tools: ToolsConfig,
        *,
        fail_fast: bool = False,
    ) -> None:
        """Initialize the build executor.

        Parameters
        ----------
        graph
            Target graph containing all registered targets.
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        paths
            Build paths configuration.
        tools
            Tools configuration for build execution.
        fail_fast
            If True, stop on first error. If False (default), continue
            executing independent targets and collect all errors.
        """
        self._graph = graph
        self._gateway = gateway
        self._snapshot = snapshot
        self._paths = paths
        self._tools = tools
        self._fail_fast = fail_fast
        self._export_options: ExportCallOptions | None = None
        # Initialize providers for protocol-based DI
        self._providers: Providers = create_default_providers(tools)
        # Load build config for target parameters
        self._config: BuildConfig = load_build_config(snapshot.repo_root)

    @property
    def export_options(self) -> ExportCallOptions | None:
        """Get export options for export stage execution.

        Returns
        -------
        ExportCallOptions | None
            Export options or None if not configured.
        """
        return self._export_options

    @export_options.setter
    def export_options(self, value: ExportCallOptions | None) -> None:
        """Set export options for export stage execution.

        Parameters
        ----------
        value
            Export options (validation, dataset selection, etc.).
        """
        self._export_options = value

    @property
    def providers(self) -> Providers:
        """Get the DI providers for external tools.

        Returns
        -------
        Providers
            Container with all protocol implementations.
        """
        return self._providers

    @property
    def config(self) -> BuildConfig:
        """Get the build configuration.

        Returns
        -------
        BuildConfig
            Configuration loaded from codeintel.build.toml.
        """
        return self._config

    def execute(
        self,
        plan: BuildPlan,
        *,
        dry_run: bool = False,
    ) -> BuildResult:
        """Execute a build plan.

        Runs all stages in the plan sequentially, recording manifests
        for completed targets and handling failures gracefully.

        Parameters
        ----------
        plan
            The build plan to execute.
        dry_run
            If True, validate and return plan info without executing.

        Returns
        -------
        BuildResult
            Execution result with completed, failed, and skipped targets.

        Examples
        --------
        >>> result = executor.execute(plan)
        >>> if result.success:
        ...     print(f"Built {len(result.completed_targets)} targets")
        """
        start_time = datetime.now(tz=UTC)
        run_id = self._generate_run_id()

        log.info(
            "build.executor.start run_id=%s targets=%d stages=%d dry_run=%s",
            run_id,
            plan.total_steps,
            len(plan.stages),
            dry_run,
        )

        # Phase A: Start run tracking
        self._start_run(run_id, plan)

        # Phase B: Handle dry run
        if dry_run:
            return self._complete_dry_run(run_id, plan, start_time)

        # Phase C: Execute stages with continue-and-collect semantics
        completed: list[str] = []
        failed: list[str] = []
        error_collection = BuildErrorCollection()
        status: BuildStatus = "succeeded"

        try:
            for stage in plan.stages:
                stage_result = self._execute_stage(stage, run_id)

                # Record manifests for completed targets
                for target_name in stage_result.completed:
                    self._record_manifest(
                        target_name=target_name,
                        duration_ms=stage_result.durations_ms.get(target_name, 0.0),
                        row_count=stage_result.row_counts.get(target_name),
                    )

                completed.extend(stage_result.completed)
                failed.extend(stage_result.failed)

                # Collect errors for failed targets
                if stage_result.error:
                    for target_name in stage_result.failed:
                        error_collection.add(
                            PluginExecutionError(
                                target=target_name,
                                plugin=self._graph.get(target_name).plugin,
                                cause=Exception(stage_result.error),
                            )
                        )

                # Check fail-fast mode
                if not stage_result.success:
                    status = "failed"
                    if self._fail_fast:
                        log.info(
                            "build.executor.fail_fast run_id=%s failed_targets=%s",
                            run_id,
                            stage_result.failed,
                        )
                        break
                    # Continue-and-collect: keep going with other stages
                    log.info(
                        "build.executor.continue_after_failure run_id=%s failed=%s continuing=True",
                        run_id,
                        stage_result.failed,
                    )

        except Exception as exc:
            log.exception("build.executor.error run_id=%s", run_id)
            status = "failed"
            error_collection.add(
                PluginExecutionError(
                    target="<executor>",
                    plugin="<executor>",
                    cause=exc,
                )
            )

        # Generate error summary from collection
        error_summary: str | None = None
        if error_collection.has_errors:
            error_summary = f"{len(error_collection)} error(s) during build"

        # Phase D: Complete run tracking
        duration_ms = (datetime.now(tz=UTC) - start_time).total_seconds() * 1000
        self._complete_run(
            run_id=run_id,
            status=status,
            completed=tuple(completed),
            skipped=plan.skipped_targets,
            error_summary=error_summary,
        )

        log.info(
            "build.executor.complete run_id=%s status=%s completed=%d failed=%d",
            run_id,
            status,
            len(completed),
            len(failed),
        )

        return BuildResult(
            run_id=run_id,
            plan=plan,
            status=status,
            completed_targets=tuple(completed),
            failed_targets=tuple(failed),
            skipped_targets=plan.skipped_targets,
            duration_ms=duration_ms,
            error_summary=error_summary,
            errors=error_collection,
        )

    # =========================================================================
    # Run ID Generation
    # =========================================================================

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID.

        Returns
        -------
        str
            Run ID in format "build-YYYYMMDD-HHMMSS-xxxxxxxx".
        """
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"build-{timestamp}-{suffix}"

    # =========================================================================
    # Run Tracking
    # =========================================================================

    def _start_run(self, run_id: str, plan: BuildPlan) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        run_id
            Unique run identifier.
        plan
            Build plan being executed.
        """
        record = BuildRunRecord(
            run_id=run_id,
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            requested_targets=plan.requested_targets,
            computed_targets=(),
            skipped_targets=plan.skipped_targets,
            started_at=datetime.now(tz=UTC),
            status="running",
        )
        self._gateway.build.start_run(record)

    def _complete_run(
        self,
        run_id: str,
        status: BuildStatus,
        completed: tuple[str, ...],
        skipped: tuple[str, ...],
        error_summary: str | None,
    ) -> None:
        """Record completion of a build run.

        Parameters
        ----------
        run_id
            Unique run identifier.
        status
            Final status (succeeded or failed).
        completed
            Targets that were computed.
        skipped
            Targets that were skipped.
        error_summary
            Error summary if failed.
        """
        self._gateway.build.complete_run(
            run_id=run_id,
            status=status,
            computed_targets=completed,
            skipped_targets=skipped,
            error_summary=error_summary,
        )

    def _complete_dry_run(
        self,
        run_id: str,
        plan: BuildPlan,
        start_time: datetime,
    ) -> BuildResult:
        """Complete a dry run without executing.

        Parameters
        ----------
        run_id
            Unique run identifier.
        plan
            Build plan that would be executed.
        start_time
            When the run started.

        Returns
        -------
        BuildResult
            Result indicating what would have been executed.
        """
        duration_ms = (datetime.now(tz=UTC) - start_time).total_seconds() * 1000

        # Mark run as succeeded (dry run is always successful)
        self._complete_run(
            run_id=run_id,
            status="succeeded",
            completed=(),
            skipped=plan.skipped_targets,
            error_summary=None,
        )

        # Collect all targets that would be computed
        would_compute = tuple(step.target for stage in plan.stages for step in stage.steps)

        log.info(
            "build.executor.dry_run run_id=%s would_compute=%d",
            run_id,
            len(would_compute),
        )

        return BuildResult(
            run_id=run_id,
            plan=plan,
            status="succeeded",
            completed_targets=(),
            failed_targets=(),
            skipped_targets=plan.skipped_targets,
            duration_ms=duration_ms,
            error_summary=None,
        )

    # =========================================================================
    # Manifest Recording
    # =========================================================================

    def _record_manifest(
        self,
        target_name: str,
        duration_ms: float,
        row_count: int | None,
    ) -> None:
        """Record a manifest for a completed target.

        Parameters
        ----------
        target_name
            Name of the target that was computed.
        duration_ms
            Execution duration in milliseconds.
        row_count
            Number of rows written, or None if unknown.
        """
        target = self._graph.get(target_name)
        input_hash = compute_input_hash(target, self._snapshot, self._gateway)

        manifest = OutputManifest(
            target=target_name,
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            plugin=target.plugin,
            computed_at=datetime.now(tz=UTC),
            duration_ms=duration_ms,
            input_hash=input_hash,
            row_count=row_count,
        )
        self._gateway.build.save_manifest(manifest)

        log.debug(
            "build.executor.manifest target=%s input_hash=%s",
            target_name,
            input_hash,
        )

    # =========================================================================
    # Direct Plugin Execution
    # =========================================================================

    def _execute_target_direct(
        self,
        target_name: str,
    ) -> tuple[bool, str | None, dict[str, int]]:
        """Execute a target directly via the plugin registry.

        This method bypasses the legacy domain executors and calls plugins
        directly using the unified TargetPlugin interface.

        Parameters
        ----------
        target_name
            Name of the target to execute.

        Returns
        -------
        tuple[bool, str | None, dict[str, int]]
            (success, error_message, row_counts)
        """
        try:
            # Get target and plugin
            target = self._graph.get(target_name)
            plugin = get_plugin_for_target(target_name)

            # Build execution context
            resources = ContextResources(
                providers=self._providers,
                gateway=self._gateway,
                modules=(),  # Will be loaded from DB if needed
            )

            # Get parameters from config
            params = self._config.parameters_for(target_name)

            ctx = TargetExecutionContext(
                target=target,
                snapshot=self._snapshot,
                paths=self._paths,
                resources=resources,
                parameters=params,
            )

            # Execute plugin
            result = asyncio.run(plugin.execute(ctx))

        except KeyError as e:
            # No plugin registered for this target
            log.warning("No plugin for target '%s', falling back to legacy", target_name)
            return False, f"No plugin registered: {e}", {}
        except Exception as e:
            log.exception("Direct plugin execution failed for %s", target_name)
            return False, str(e), {}
        else:
            if result.success:
                return True, None, dict(result.row_counts)
            return False, result.error_message, {}

    # =========================================================================
    # Stage Execution
    # =========================================================================

    def _execute_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute a single build stage.

        Dispatches to the appropriate module executor based on stage type.

        Parameters
        ----------
        stage
            Plan stage to execute.
        run_id
            Current run identifier.

        Returns
        -------
        StageExecutionResult
            Result with completed and failed targets.
        """
        log.info(
            "build.executor.stage.start module=%s steps=%d",
            stage.module,
            stage.step_count,
        )

        if stage.module == "ingestion":
            return self._execute_ingestion_stage(stage, run_id)
        if stage.module == "graphs":
            return self._execute_graphs_stage(stage, run_id)
        if stage.module == "analytics":
            return self._execute_analytics_stage(stage, run_id)
        if stage.module == "export":
            return self._execute_export_stage(stage, run_id)

        # Should never happen - TargetModule is a literal type
        message = f"Unknown stage module: {stage.module}"
        return StageExecutionResult(
            module=stage.module,
            completed=(),
            failed=tuple(step.target for step in stage.steps),
            error=message,
        )

    def _execute_ingestion_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute an ingestion stage.

        Executes each ingestion target directly via the plugin registry.

        Parameters
        ----------
        stage
            Ingestion stage to execute.
        run_id
            Current run identifier.

        Returns
        -------
        StageExecutionResult
            Result with completed and failed targets.
        """
        start_time = datetime.now(tz=UTC)
        target_names = [step.target for step in stage.steps]

        log.debug(
            "build.executor.ingestion targets=%s run_id=%s",
            target_names,
            run_id,
        )

        completed: list[str] = []
        failed: list[str] = []
        durations_ms: dict[str, float] = {}
        row_counts: dict[str, int | None] = {}
        errors: list[str] = []

        for target_name in target_names:
            target_start = datetime.now(tz=UTC)
            success, error, counts = self._execute_target_direct(target_name)
            duration = (datetime.now(tz=UTC) - target_start).total_seconds() * 1000

            durations_ms[target_name] = duration
            row_counts.update(counts)

            if success:
                completed.append(target_name)
                log.info(
                    "build.executor.ingestion.target.complete target=%s duration_ms=%.1f",
                    target_name,
                    duration,
                )
            else:
                failed.append(target_name)
                if error:
                    errors.append(f"{target_name}: {error}")
                log.warning(
                    "build.executor.ingestion.target.failed target=%s error=%s",
                    target_name,
                    error,
                )

        total_duration = (datetime.now(tz=UTC) - start_time).total_seconds() * 1000
        log.info(
            "build.executor.ingestion.complete completed=%d failed=%d duration_ms=%.1f",
            len(completed),
            len(failed),
            total_duration,
        )

        return StageExecutionResult(
            module="ingestion",
            completed=tuple(completed),
            failed=tuple(failed),
            durations_ms=durations_ms,
            row_counts=row_counts,
            error="; ".join(errors) if errors else None,
        )

    def _execute_graphs_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute a graphs stage.

        Maps stage targets to graph plugins and executes via graph executor.

        Parameters
        ----------
        stage
            Graphs stage to execute.
        run_id
            Current run identifier.

        Returns
        -------
        StageExecutionResult
            Result with completed and failed targets.
        """
        target_names = [step.target for step in stage.steps]
        plugin_names = self._get_plugin_names_for_stage(stage)

        log.debug(
            "build.executor.graphs targets=%s plugins=%s",
            target_names,
            plugin_names,
        )

        try:
            # Create planning context with available fields
            # Note: GraphPlanContext uses cfg/runtime_snapshot/target for inputs
            graph_policy = GraphPluginPolicy()
            plan_context = GraphPlanContext(
                runtime_snapshot=self._snapshot,
                target=(self._snapshot.repo, self._snapshot.commit),
                policy=graph_policy,
            )

            # Plan the graph plugin run
            plan = plan_graph_plugin_run(
                plugin_names=plugin_names if plugin_names else None,
                context=plan_context,
            )

            # Create execution context and executor
            exec_context = GraphExecutorContext(
                gateway=self._gateway,
                snapshot=self._snapshot,
            )

            # Convert GraphPluginPolicy to BaseExecutionPolicy
            base_policy = BaseExecutionPolicy(
                fail_fast=graph_policy.fail_fast,
                default_severity=graph_policy.default_severity,
                severity_overrides=graph_policy.severity_overrides,
                skip_on_unchanged=graph_policy.skip_on_unchanged,
                dry_run=graph_policy.dry_run,
                timeouts_by_plugin=graph_policy.timeouts_ms,
            )

            # Execute using the new GraphPluginExecutor
            executor = GraphPluginExecutor(
                policy=base_policy,
                prior_manifest=plan.prior_manifest,
                scope=plan.scope,
            )

            report = executor.execute(
                executor_ctx=exec_context,
                plugins=plan.plugins,
                run_id=plan.run_id,
                settings_by_plugin=plan.settings_by_plugin,
            )

            # Extract results from report
            completed: list[str] = []
            failed: list[str] = []
            durations: dict[str, float] = {}
            row_counts: dict[str, int | None] = {}

            for record in report.records:
                # Map plugin name back to target name
                target_name = self._plugin_to_target(record.plugin_name, target_names)
                if target_name is None:
                    continue

                if record.status == "succeeded":
                    completed.append(target_name)
                    durations[target_name] = record.duration_ms
                    # PluginExecutionRecord doesn't have row_count; use None
                    row_counts[target_name] = None
                else:
                    failed.append(target_name)

            error: str | None = None
            if report.fatal_error:
                error = "Graph execution encountered a fatal error"

            return StageExecutionResult(
                module="graphs",
                completed=tuple(completed),
                failed=tuple(failed),
                durations_ms=durations,
                row_counts=row_counts,
                error=error,
            )

        except Exception as exc:
            log.exception("build.executor.graphs.error run_id=%s", run_id)
            return StageExecutionResult(
                module="graphs",
                completed=(),
                failed=tuple(target_names),
                error=str(exc),
            )

    def _execute_analytics_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute an analytics stage.

        Executes analytics plugins directly via the TargetPlugin interface,
        similar to ingestion targets.

        Parameters
        ----------
        stage
            Analytics stage to execute.
        run_id
            Current run identifier.

        Returns
        -------
        StageExecutionResult
            Result with completed and failed targets.
        """
        log.debug(
            "build.executor.analytics targets=%s run_id=%s",
            [step.target for step in stage.steps],
            run_id,
        )

        # Build plugin lookup from ALL_PLUGINS
        plugin_lookup = {p.plugin_name: p for p in ALL_PLUGINS}

        completed: list[str] = []
        failed: list[str] = []
        durations: dict[str, float] = {}
        row_counts: dict[str, int | None] = {}

        for step in stage.steps:
            success, duration_ms, counts = self._execute_analytics_target(
                step.target,
                step.plugin,
                plugin_lookup,
            )
            if success:
                completed.append(step.target)
                if duration_ms is not None:
                    durations[step.target] = duration_ms
                if counts:
                    row_counts[step.target] = sum(counts.values())
            else:
                failed.append(step.target)

        return StageExecutionResult(
            module="analytics",
            completed=tuple(completed),
            failed=tuple(failed),
            durations_ms=durations,
            row_counts=row_counts,
            error=None if not failed else f"Failed targets: {', '.join(failed)}",
        )

    def _execute_analytics_target(
        self,
        target_name: str,
        plugin_name: str,
        plugin_lookup: dict[str, Any],
    ) -> tuple[bool, float | None, dict[str, int]]:
        """Execute a single analytics target.

        Parameters
        ----------
        target_name
            Name of the target to execute.
        plugin_name
            Name of the plugin to use.
        plugin_lookup
            Mapping of plugin names to plugin instances.

        Returns
        -------
        tuple[bool, float | None, dict[str, int]]
            (success, duration_ms, row_counts)
        """
        plugin = plugin_lookup.get(plugin_name)
        if plugin is None:
            log.warning(
                "No analytics plugin found for '%s' (plugin=%s)",
                target_name,
                plugin_name,
            )
            return False, None, {}

        start_time = datetime.now(tz=UTC)
        try:
            target = self._graph.get(target_name)
            resources = ContextResources(
                providers=self._providers,
                gateway=self._gateway,
                modules=(),
            )
            ctx = TargetExecutionContext(
                target=target,
                snapshot=self._snapshot,
                paths=self._paths,
                resources=resources,
                parameters=self._config.parameters_for(target_name),
            )

            result = asyncio.run(plugin.execute(ctx))
        except Exception:
            log.exception(
                "build.executor.analytics.target.error target=%s",
                target_name,
            )
            return False, None, {}

        duration_ms = (datetime.now(tz=UTC) - start_time).total_seconds() * 1000

        if result.success:
            log.info(
                "build.executor.analytics.target.success target=%s duration_ms=%.1f",
                target_name,
                duration_ms,
            )
            return True, duration_ms, dict(result.row_counts)

        log.warning(
            "build.executor.analytics.target.failed target=%s error=%s",
            target_name,
            result.error_message,
        )
        return False, duration_ms, {}

    def _execute_export_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute an export stage.

        Export stages produce files (JSONL, Parquet) rather than database tables.
        Each export target is executed independently.

        Parameters
        ----------
        stage
            Export stage to execute.
        run_id
            Current run identifier.

        Returns
        -------
        StageExecutionResult
            Result with completed and failed targets.
        """
        start_time = datetime.now(tz=UTC)
        target_names = [step.target for step in stage.steps]
        completed: list[str] = []
        failed: list[str] = []
        durations: dict[str, float] = {}

        log.debug(
            "build.executor.export targets=%s",
            target_names,
        )

        document_output_dir = self._paths.document_output_dir

        for target_name in target_names:
            target_start = datetime.now(tz=UTC)
            try:
                if target_name == "export_jsonl":
                    export_all_jsonl(
                        self._gateway,
                        document_output_dir,
                        options=self._export_options,
                    )
                    completed.append(target_name)
                elif target_name == "export_parquet":
                    export_all_parquet(
                        self._gateway,
                        document_output_dir,
                        options=self._export_options,
                    )
                    completed.append(target_name)
                else:
                    log.warning(
                        "build.executor.export.unknown_target target=%s",
                        target_name,
                    )
                    failed.append(target_name)
                    continue

                duration_ms = int((datetime.now(tz=UTC) - target_start).total_seconds() * 1000)
                durations[target_name] = duration_ms
                log.info(
                    "build.executor.export.completed target=%s duration_ms=%d",
                    target_name,
                    duration_ms,
                )

            except Exception:
                log.exception(
                    "build.executor.export.failed target=%s run_id=%s",
                    target_name,
                    run_id,
                )
                failed.append(target_name)
                durations[target_name] = int(
                    (datetime.now(tz=UTC) - target_start).total_seconds() * 1000
                )

        total_duration_ms = int((datetime.now(tz=UTC) - start_time).total_seconds() * 1000)
        log.info(
            "build.executor.export.stage_complete completed=%d failed=%d duration_ms=%d",
            len(completed),
            len(failed),
            total_duration_ms,
        )

        return StageExecutionResult(
            module="export",
            completed=tuple(completed),
            failed=tuple(failed),
            durations_ms=durations,
            row_counts={},  # Exports produce files, not rows
            error=None if not failed else f"Failed targets: {', '.join(failed)}",
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_plugin_names_for_stage(self, stage: PlanStage) -> list[str]:
        """Get plugin names for all targets in a stage.

        Parameters
        ----------
        stage
            Stage to get plugin names for.

        Returns
        -------
        list[str]
            Plugin names for each target in the stage.
        """
        return [self._graph.get(step.target).plugin for step in stage.steps]

    def _plugin_to_target(
        self,
        plugin_name: str,
        target_names: list[str],
    ) -> str | None:
        """Map a plugin name back to a target name.

        Parameters
        ----------
        plugin_name
            Plugin name from execution report.
        target_names
            Expected target names for this stage.

        Returns
        -------
        str | None
            Target name if found, None otherwise.
        """
        for target_name in target_names:
            target = self._graph.get(target_name)
            if target.plugin == plugin_name:
                return target_name
        return None


__all__ = [
    "BuildExecutor",
    "BuildResult",
    "StageExecutionResult",
]
