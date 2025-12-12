"""Build handlers.

Handlers for build operations, status, and history.
"""

from __future__ import annotations

# Used by build_plan_handler for JSON serialization
import json as _json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.build.config import load_build_config
from codeintel.build.executor import BuildExecutor, ExecutorEnv
from codeintel.build.hamilton import BuildEnv, HamiltonBuildExecutor
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.plan import PlanGenerator
from codeintel.build.providers import create_default_providers
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    BuildExplainResult,
    BuildHistoryResult,
    BuildPlanResult,
    BuildRunResult,
    BuildStatusResult,
)
from codeintel.cli.errors import ValidationError
from codeintel.cli.errors.results import (
    fail_build_run_not_found,
    fail_execution_failed,
    fail_invalid_module,
    fail_invalid_target_selection,
    fail_invalid_targets,
    fail_project_error,
)
from codeintel.cli.handlers._utilities import runtime_gateway
from codeintel.cli.resolution.errors import ResolutionError

if TYPE_CHECKING:
    from codeintel.build.executor import BuildResult
    from codeintel.build.hamilton import HamiltonBuildResult
    from codeintel.build.manifest import BuildRunRecord
    from codeintel.build.plan import BuildPlan
    from codeintel.build.targets import TargetGraph, TargetModule
    from codeintel.cli.context import CommandContext
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


class BuildResultLike(Protocol):
    """Protocol for build result objects."""

    @property
    def completed_targets(self) -> tuple[str, ...]:
        """Return targets that completed successfully."""
        ...

    @property
    def skipped_targets(self) -> tuple[str, ...]:
        """Return targets that were skipped."""
        ...

    @property
    def failed_targets(self) -> tuple[str, ...]:
        """Return targets that failed."""
        ...

    @property
    def duration_ms(self) -> float:
        """Return total duration in milliseconds."""
        ...


class RunMode(Enum):
    """Build execution mode."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class TargetScope(Enum):
    """Scope selector for build goals."""

    REQUESTED = "requested"
    ALL = "all"


def _resolve_goals(
    targets: list[str] | None,
    module: str | None,
    target_scope: TargetScope,
    graph: TargetGraph,
) -> list[str]:
    """Resolve target goals from CLI arguments.

    Parameters
    ----------
    targets
        Explicit target names from CLI arguments.
    module
        Module name to build all targets for.
    target_scope
        Scope selector indicating whether to build all targets or only requested ones.
    graph
        Target graph for validation.

    Returns
    -------
    list[str]
        Resolved target names.

    Raises
    ------
    ValidationError
        If no targets specified or unknown target provided.
    """
    if target_scope is TargetScope.ALL:
        return [t.name for t in graph.all_targets]

    if module:
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            msg = f"Unknown module: {module}. Valid: {', '.join(valid_modules)}"
            raise ValidationError(msg)
        module_typed = cast("TargetModule", module)
        module_targets = graph.targets_for_module(module_typed)
        return [t.name for t in module_targets]

    if targets:
        for target in targets:
            try:
                graph.get(target)
            except KeyError as exc:
                msg = f"Unknown target: {target}"
                raise ValidationError(msg) from exc
        return list(targets)

    msg = "Specify targets, --module, or --all"
    raise ValidationError(msg)


def _group_targets_by_status(
    state: DatabaseState,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Group targets by their status.

    Parameters
    ----------
    state
        Database state from StateValidator.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        (computed, stale, missing, blocked) lists.
    """
    computed: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    blocked: list[str] = []

    for target_name, target_state in state.targets.items():
        if target_state.status == "computed":
            computed.append(target_name)
        elif target_state.status == "stale":
            reason = (
                f" ({target_state.staleness_reason.details})"
                if target_state.staleness_reason
                else ""
            )
            stale.append(f"{target_name}{reason}")
        elif target_state.status == "missing":
            missing.append(target_name)
        elif target_state.status == "blocked":
            reason = (
                f" ({target_state.staleness_reason.details})"
                if target_state.staleness_reason
                else ""
            )
            blocked.append(f"{target_name}{reason}")

    return computed, stale, missing, blocked


def _build_status_result(state: DatabaseState) -> BuildStatusResult:
    """Build status result from database state.

    Parameters
    ----------
    state
        Database state from validator.

    Returns
    -------
    BuildStatusResult
        Status result with counts.
    """
    targets: list[dict[str, object]] = []

    computed, stale_list, missing_list, blocked_list = _group_targets_by_status(state)

    targets.extend({"name": name, "status": "fresh"} for name in computed)
    targets.extend({"name": name, "status": "stale"} for name in stale_list)
    targets.extend({"name": name, "status": "missing"} for name in missing_list)
    targets.extend({"name": name, "status": "blocked"} for name in blocked_list)

    stale_count = len(stale_list) + len(missing_list)

    return BuildStatusResult(
        targets=targets,
        stale_count=stale_count,
        fresh_count=len(computed),
    )


def _execute_build(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    goals: list[str],
    force_targets: list[str] | None,
    run_mode: RunMode,
) -> tuple[BuildResult | None, BuildPlan]:
    """Execute build with the full build pipeline.

    Parameters
    ----------
    runtime
        Resolved runtime context.
    gateway
        Open storage gateway.
    goals
        Target names to build.
    force_targets
        Targets to force recompute.
    run_mode
        Whether to execute or return a dry-run plan.

    Returns
    -------
    tuple[BuildResult | None, BuildPlan]
        (result, plan) tuple. result is None for dry-run.
    """
    graph = get_target_graph()

    LOG.info("build.cli.validate_state goals=%s", goals)
    validator = StateValidator(graph, gateway, runtime.snapshot)
    state = validator.validate()

    LOG.info("build.cli.resolve force=%s", force_targets)
    resolver = BuildResolver(graph, state)
    resolution = resolver.resolve(goals, force_recompute=tuple(force_targets or ()))

    LOG.info("build.cli.generate_plan to_compute=%d", len(resolution.to_compute))
    generator = PlanGenerator(graph)
    plan = generator.generate(resolution)

    if run_mode is RunMode.DRY_RUN:
        LOG.info("build.cli.dry_run stages=%d", len(plan.stages))
        return None, plan

    LOG.info("build.cli.execute stages=%d", len(plan.stages))
    env = ExecutorEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        tools=runtime.tools,
    )
    executor = BuildExecutor(graph=graph, env=env)
    result = executor.execute(plan)
    return result, plan


def _execute_build_hamilton(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    goals: list[str],
    run_mode: RunMode,
    force: list[str] | None = None,
    hamilton_mode: str = "generated",
) -> BuildResultLike | None:
    """Execute build using Hamilton executor.

    Parameters
    ----------
    runtime
        Resolved runtime context.
    gateway
        Open storage gateway.
    goals
        Target names to build.
    run_mode
        Whether to execute or return a dry-run plan.
    force
        Optional list of targets to force recompute.
    hamilton_mode
        Hamilton node mode: "generated" (default) or "phase0" (debug).

    Returns
    -------
    BuildResult | None
        Build result or None for dry-run.
    """
    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode

    if run_mode is RunMode.DRY_RUN:
        LOG.info("build.cli.hamilton.dry_run goals=%s", goals)
        return None

    LOG.info(
        "build.cli.hamilton.execute goals=%s force=%s mode=%s",
        goals,
        force,
        hamilton_mode,
    )

    # Build the Hamilton environment
    providers = create_default_providers(runtime.tools)
    config = load_build_config(runtime.snapshot.repo_root)

    # Prefetch all manifests for this repo/commit to avoid per-target DB calls
    manifests_list = gateway.build.list_manifests(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
    )
    manifest_index = {m.target: m for m in manifests_list}
    LOG.debug("build.cli.hamilton.manifest_index count=%d", len(manifest_index))

    env = BuildEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        providers=providers,
        config=config,
        profile="default",
        force_targets=frozenset(force or ()),
        manifest_index=manifest_index,
    )

    # Cast hamilton_mode to proper type - validated earlier
    mode: HamiltonNodeMode = "generated" if hamilton_mode == "generated" else "phase0"
    executor = HamiltonBuildExecutor(profile="default", mode=mode)
    hamilton_result = executor.run(env=env, targets=goals)

    # Convert Hamilton result to BuildResult-like object for compatibility
    # We create a minimal adapter that has the expected attributes
    return _HamiltonResultAdapter(hamilton_result)


class _HamiltonResultAdapter:
    """Adapter to make HamiltonBuildResult compatible with BuildResult interface."""

    def __init__(self, hamilton_result: HamiltonBuildResult) -> None:
        """Initialize adapter with Hamilton result.

        Parameters
        ----------
        hamilton_result
            Result from HamiltonBuildExecutor.
        """
        self._result = hamilton_result

    @property
    def completed_targets(self) -> tuple[str, ...]:
        """Return targets that completed successfully.

        Uses the new computed_targets field from HamiltonBuildResult.
        """
        return self._result.computed_targets

    @property
    def skipped_targets(self) -> tuple[str, ...]:
        """Return targets that were skipped.

        Uses the new skipped_targets field from HamiltonBuildResult.
        """
        return self._result.skipped_targets

    @property
    def failed_targets(self) -> tuple[str, ...]:
        """Return targets that failed."""
        return self._result.failed_targets

    @property
    def duration_ms(self) -> float:
        """Return total duration in milliseconds.

        Uses the duration_ms field from HamiltonBuildResult.
        """
        return self._result.duration_ms


def _lookup_run_by_id(
    gateway: StorageGateway,
    repo: str,
    run_id: str,
) -> BuildRunRecord:
    """Look up a build run by ID or prefix.

    Parameters
    ----------
    gateway
        Open storage gateway.
    repo
        Repository slug for filtering runs.
    run_id
        Exact run ID or prefix to match.

    Returns
    -------
    BuildRunRecord
        The matched run record.

    Raises
    ------
    ValidationError
        If run is not found or prefix is ambiguous.
    """
    record = gateway.build.fetch_run(run_id)
    if record is not None:
        return record

    all_runs = gateway.build.list_runs(repo=repo, limit=100)
    matches = [r for r in all_runs if r.run_id.startswith(run_id)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        msg = f"Ambiguous run ID prefix '{run_id}' matches {len(matches)} runs"
        raise ValidationError(msg)

    msg = f"Run not found: {run_id}"
    raise ValidationError(msg)


def build_status_handler(
    ctx: CommandContext,
) -> CliResult[BuildStatusResult]:
    """Show current state of all build targets.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - module: Optional module filter (ingestion, graphs, analytics).

    Returns
    -------
    CliResult[BuildStatusResult]
        Structured result with target status information.
    """
    module = ctx.params.get_str("module")

    try:
        runtime = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()

    LOG.info(
        "build.status repo=%s commit=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
    )

    gateway = ctx.gateway
    validator = StateValidator(graph, gateway, runtime.snapshot)
    state = validator.validate()

    if module:
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            return fail_invalid_module(module, valid_modules)

        module_targets = graph.targets_for_module(cast("TargetModule", module))
        module_names = {t.name for t in module_targets}
        state = DatabaseState(
            repo=state.repo,
            commit=state.commit,
            targets={
                name: target_state
                for name, target_state in state.targets.items()
                if name in module_names
            },
        )

    return CliResult.ok(_build_status_result(state))


@dataclass
class _BuildRunParams:
    """Extracted build run parameters."""

    targets: list[str] | None
    module: str | None
    all_targets: bool
    dry_run: bool
    force: list[str] | None
    engine: str
    hamilton_mode: str


def _extract_build_run_params(ctx: CommandContext) -> _BuildRunParams:
    """Extract and normalize build run parameters from context.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    _BuildRunParams
        Extracted parameters.
    """
    targets_list = ctx.params.get_list("targets")
    targets: list[str] | None = targets_list if targets_list else None

    force_list = ctx.params.get_list("force")
    force: list[str] | None = force_list if force_list else None

    return _BuildRunParams(
        targets=targets,
        module=ctx.params.get_str("module"),
        all_targets=ctx.params.get_bool("all_targets"),
        dry_run=ctx.params.get_bool("dry_run"),
        force=force,
        engine=ctx.params.get_str("engine") or "hamilton",
        hamilton_mode=ctx.params.get_str("hamilton_mode") or "generated",
    )


def _validate_build_run_params(
    params: _BuildRunParams,
) -> CliResult[BuildRunResult] | None:
    """Validate build run parameters.

    Parameters
    ----------
    params
        Build run parameters.

    Returns
    -------
    CliResult[BuildRunResult] | None
        Error result if validation fails, None if valid.
    """
    if params.module:
        valid_modules = ("ingestion", "graphs", "analytics")
        if params.module not in valid_modules:
            return fail_invalid_module(params.module, valid_modules)

    provided = [bool(params.targets), params.module is not None, params.all_targets]
    if sum(provided) != 1:
        return fail_invalid_target_selection("Provide exactly one of targets, --module, or --all.")

    # Validate engine parameter
    valid_engines = ("legacy", "hamilton")
    if params.engine not in valid_engines:
        return fail_invalid_target_selection(
            f"Invalid engine '{params.engine}'. Valid: {', '.join(valid_engines)}"
        )

    # Validate hamilton_mode parameter
    valid_modes = ("phase0", "generated")
    if params.hamilton_mode not in valid_modes:
        return fail_invalid_target_selection(
            f"Invalid hamilton_mode '{params.hamilton_mode}'. Valid: {', '.join(valid_modes)}"
        )

    return None


def build_run_handler(
    ctx: CommandContext,
) -> CliResult[BuildRunResult]:
    """Build targets with automatic dependency resolution.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - targets: Target names to build.
        - module: Module name (ingestion, graphs, analytics).
        - all_targets: Build all targets.
        - dry_run: Show plan without executing.
        - force: Force recompute of specific targets.

    Returns
    -------
    CliResult[BuildRunResult]
        Structured result with build execution information.
    """
    params = _extract_build_run_params(ctx)
    validation_error = _validate_build_run_params(params)
    if validation_error is not None:
        return validation_error

    try:
        runtime = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()
    scope = TargetScope.ALL if params.all_targets else TargetScope.REQUESTED

    try:
        goals = _resolve_goals(params.targets, params.module, scope, graph)
    except ValidationError as e:
        return fail_invalid_targets(str(e))

    LOG.info(
        "build.run repo=%s commit=%s targets=%s engine=%s hamilton_mode=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        goals,
        params.engine,
        params.hamilton_mode,
    )

    run_mode = RunMode.DRY_RUN if params.dry_run else RunMode.EXECUTE
    return _execute_and_format_result(
        runtime, goals, params.force, run_mode, params.engine, params.hamilton_mode
    )


def _execute_and_format_result(
    runtime: ResolvedRuntime,
    goals: list[str],
    force: list[str] | None,
    run_mode: RunMode,
    engine: str = "hamilton",
    hamilton_mode: str = "generated",
) -> CliResult[BuildRunResult]:
    """Execute build and format result.

    Parameters
    ----------
    runtime
        Resolved runtime.
    goals
        Target goals.
    force
        Force recompute targets.
    run_mode
        Execution mode.
    engine
        Build engine to use: "hamilton" (default) or "legacy".
    hamilton_mode
        Hamilton node mode: "generated" (default) or "phase0" (debug).

    Returns
    -------
    CliResult[BuildRunResult]
        Build result.
    """
    try:
        with runtime_gateway(runtime, read_only=False) as gateway:
            if engine == "hamilton":
                result = _execute_build_hamilton(
                    runtime, gateway, goals, run_mode, force, hamilton_mode
                )
            else:
                result, _plan = _execute_build(runtime, gateway, goals, force, run_mode)
    except Exception as exc:
        LOG.exception("build.run.error engine=%s", engine)
        return fail_execution_failed("build", str(exc))

    if run_mode is RunMode.DRY_RUN or result is None:
        return CliResult.ok(
            BuildRunResult(
                executed=[],
                skipped=[],
                failed=[],
                duration_seconds=0.0,
            )
        )

    return CliResult.ok(
        BuildRunResult(
            executed=list(result.completed_targets),
            skipped=list(result.skipped_targets),
            failed=list(result.failed_targets),
            duration_seconds=result.duration_ms / 1000.0,
        )
    )


def build_history_handler(
    ctx: CommandContext,
) -> CliResult[BuildHistoryResult]:
    """Show build run history and details.

    Parameters
    ----------
    ctx
        Command context with params:
        - project_root: Optional project root override.
        - run_id: Optional specific run ID to show.
        - limit: Maximum number of runs to show (default 10).

    Returns
    -------
    CliResult[BuildHistoryResult]
        Structured result with build history.
    """
    run_id = ctx.params.get_str("run_id")
    limit = ctx.params.get_int("limit", 10)

    try:
        runtime = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    gateway = ctx.gateway
    if run_id:
        try:
            record = _lookup_run_by_id(gateway, runtime.snapshot.repo, run_id)
        except ValidationError as e:
            return fail_build_run_not_found(str(e))
        return CliResult.ok(
            BuildHistoryResult(
                runs=[record.to_dict()],
                count=1,
            )
        )

    runs = gateway.build.list_runs(repo=runtime.snapshot.repo, limit=limit)

    return CliResult.ok(
        BuildHistoryResult(
            runs=[r.to_dict() for r in runs],
            count=len(runs),
        )
    )


@dataclass(frozen=True)
class BuildGraphResult:
    """Result type for build graph command."""

    dag_json: str
    node_count: int
    edge_count: int


def build_graph_handler(
    ctx: CommandContext,
) -> CliResult[BuildGraphResult]:
    """Export Hamilton DAG for specified targets.

    Parameters
    ----------
    ctx
        Command context with params:
        - targets: List of target names.
        - module: Optional module filter.
        - all_targets: Whether to include all targets.
        - output_format: Output format (json or text).
        - output_file: Optional output file path.

    Returns
    -------
    CliResult[BuildGraphResult]
        Structured result with DAG information.
    """
    from pathlib import Path

    from codeintel.build.hamilton.driver_factory import build_driver
    from codeintel.build.hamilton.observability import (
        export_dag_dot,
        export_dag_json,
        export_dag_mermaid,
        get_dag_info,
    )

    # Verify we can resolve runtime (validates project)
    try:
        _ = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()

    # Parse targets
    targets_list = ctx.params.get_list("targets")
    targets: list[str] | None = targets_list if targets_list else None

    module = ctx.params.get_str("module")
    all_targets = ctx.params.get_bool("all_targets")
    output_file = ctx.params.get_str("output_file")
    output_format = ctx.params.get_str("output_format") or "json"

    try:
        goals = _resolve_goals(
            targets=targets,
            module=module,
            target_scope=TargetScope.ALL if all_targets else TargetScope.REQUESTED,
            graph=graph,
        )
    except ValidationError as e:
        return fail_invalid_target_selection(str(e))

    # Build Hamilton runtime in generated mode
    hamilton_runtime = build_driver(mode="generated")

    # Get DAG info
    dag_info = get_dag_info(hamilton_runtime, goals)

    # Export in requested format
    if output_format == "mermaid":
        dag_output = export_dag_mermaid(hamilton_runtime, goals)
    elif output_format == "dot":
        dag_output = export_dag_dot(hamilton_runtime, goals)
    else:
        dag_output = export_dag_json(hamilton_runtime, goals)

    # Write to file if specified
    if output_file:
        Path(output_file).write_text(dag_output, encoding="utf-8")
        LOG.info("build.graph.written path=%s format=%s", output_file, output_format)

    return CliResult.ok(
        BuildGraphResult(
            dag_json=dag_output,
            node_count=dag_info["node_count"],
            edge_count=dag_info["edge_count"],
        )
    )


def build_plan_handler(
    ctx: CommandContext,
) -> CliResult[BuildPlanResult]:
    """Show build plan with status and reason for each target.

    Parameters
    ----------
    ctx
        Command context with params:
        - targets: Target names to plan.
        - module: Module name (ingestion, graphs, analytics).
        - all_targets: Plan all targets.
        - force: Mark specific targets as forced.
        - output_file: Optional output file path.

    Returns
    -------
    CliResult[BuildPlanResult]
        Structured result with build plan information.
    """
    try:
        runtime = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()

    # Parse parameters
    targets_list = ctx.params.get_list("targets")
    targets: list[str] | None = targets_list if targets_list else None

    force_list = ctx.params.get_list("force")
    force: list[str] | None = force_list if force_list else None

    module = ctx.params.get_str("module")
    all_targets = ctx.params.get_bool("all_targets")
    output_file = ctx.params.get_str("output_file")

    try:
        goals = _resolve_goals(
            targets=targets,
            module=module,
            target_scope=TargetScope.ALL if all_targets else TargetScope.REQUESTED,
            graph=graph,
        )
    except ValidationError as e:
        return fail_invalid_target_selection(str(e))

    LOG.info(
        "build.plan repo=%s commit=%s targets=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        goals,
        force,
    )

    with runtime_gateway(runtime, read_only=True) as gateway:
        # Build environment for planning
        providers = create_default_providers(runtime.tools)
        config = load_build_config(runtime.snapshot.repo_root)

        # Prefetch manifests for planning
        manifests_list = gateway.build.list_manifests(
            repo=runtime.snapshot.repo,
            commit=runtime.snapshot.commit,
        )
        manifest_index = {m.target: m for m in manifests_list}

        env = BuildEnv(
            gateway=gateway,
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            providers=providers,
            config=config,
            profile="default",
            force_targets=frozenset(force or ()),
            manifest_index=manifest_index,
        )

        # Compute the plan
        plan = compute_plan(
            env=env,
            graph=graph,
            requested=tuple(goals),
            mode="generated",
        )

    # Build result
    result = BuildPlanResult(
        requested=list(plan.requested),
        closure=list(plan.closure),
        entries=[e.to_dict() for e in plan.entries],
        to_compute=list(plan.to_compute),
        to_skip=list(plan.to_skip),
        blocked=list(plan.blocked),
    )

    # Write to file if specified
    if output_file:
        Path(output_file).write_text(
            _json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        LOG.info("build.plan.written path=%s", output_file)

    return CliResult.ok(result)


def build_explain_handler(
    ctx: CommandContext,
) -> CliResult[BuildExplainResult]:
    """Explain why a target is stale and what dependencies changed.

    Parameters
    ----------
    ctx
        Command context with params:
        - target: Target name to explain.
        - force: Mark specific targets as forced.

    Returns
    -------
    CliResult[BuildExplainResult]
        Structured result with staleness explanation.
    """
    try:
        runtime = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()

    # Parse parameters
    target = ctx.params.get_str("target")
    if not target:
        return fail_invalid_targets("No target specified")

    force_list = ctx.params.get_list("force")
    force: list[str] | None = force_list if force_list else None

    # Validate target exists
    try:
        graph.get(target)
    except KeyError:
        return fail_invalid_targets(f"Unknown target: {target}")

    LOG.info(
        "build.explain repo=%s commit=%s target=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        target,
        force,
    )

    with runtime_gateway(runtime, read_only=True) as gateway:
        # Build environment for planning
        providers = create_default_providers(runtime.tools)
        config = load_build_config(runtime.snapshot.repo_root)

        # Prefetch manifests for planning
        manifests_list = gateway.build.list_manifests(
            repo=runtime.snapshot.repo,
            commit=runtime.snapshot.commit,
        )
        manifest_index = {m.target: m for m in manifests_list}

        env = BuildEnv(
            gateway=gateway,
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            providers=providers,
            config=config,
            profile="default",
            force_targets=frozenset(force or ()),
            manifest_index=manifest_index,
        )

        # Compute the plan
        plan = compute_plan(
            env=env,
            graph=graph,
            requested=(target,),
            mode="generated",
        )

    # Find the entry for the target
    entry = plan.get_entry(target)
    if entry is None:
        return fail_invalid_targets(f"Target not found in plan: {target}")

    # Generate staleness explanation
    explanation = entry.explain_staleness()

    # Build result
    result = BuildExplainResult(
        target=explanation.target,
        status=explanation.status,
        reason=explanation.reason,
        is_stale=explanation.is_stale,
        input_hash_current=explanation.input_hash_current,
        input_hash_prior=explanation.input_hash_prior,
        changed_deps=list(explanation.changed_deps),
        added_deps=list(explanation.added_deps),
        removed_deps=list(explanation.removed_deps),
        summary=explanation.summary(),
    )

    return CliResult.ok(result)


__all__ = [
    "BuildGraphResult",
    "BuildHistoryResult",
    "BuildPlanResult",
    "BuildRunResult",
    "BuildStatusResult",
    "RunMode",
    "TargetScope",
    "build_explain_handler",
    "build_graph_handler",
    "build_history_handler",
    "build_plan_handler",
    "build_run_handler",
    "build_status_handler",
]
