"""Build handlers.

Handlers for build operations, status, and history.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from codeintel.build.config import load_build_config
from codeintel.build.executor import BuildExecutor, ExecutorEnv
from codeintel.build.hamilton import BuildEnv, HamiltonBuildExecutor
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.observability import (
    export_dag_dot,
    export_dag_json,
    export_dag_mermaid,
    get_dag_info,
)
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.plan import PlanGenerator
from codeintel.build.providers import create_default_providers
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    BuildAssetsResult,
    BuildDiffResult,
    BuildExplainResult,
    BuildHistoryResult,
    BuildLineageResult,
    BuildPlanResult,
    BuildPromoteResult,
    BuildResolveResult,
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
from codeintel.storage.tracking.asset_tracking import AssetAliasRecord, AssetDiffRecord

if TYPE_CHECKING:
    from codeintel.build.executor import BuildResult
    from codeintel.build.hamilton import HamiltonBuildResult
    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode
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


@dataclass(frozen=True)
class BuildExecutionArgs:
    """Build execution options for both Hamilton and legacy engines."""

    goals: list[str]
    force: list[str] | None
    run_mode: RunMode
    engine: str
    hamilton_mode: str
    validate_outputs: bool
    strict_contracts: bool
    wrapper_allowlist: list[str] | None

    @property
    def is_dry_run(self) -> bool:
        """Return True when run_mode is DRY_RUN."""
        return self.run_mode is RunMode.DRY_RUN

    @property
    def node_mode(self) -> HamiltonNodeMode:
        """Return typed HamiltonNodeMode value."""
        return "generated" if self.hamilton_mode == "generated" else "phase0"


@dataclass(frozen=True)
class BuildPlanArgs:
    """Argument bundle for build_plan_handler."""

    targets: list[str] | None
    module: str | None
    force: list[str] | None
    all_targets: bool
    output_file: str | None


def _parse_plan_args(ctx: CommandContext) -> BuildPlanArgs:
    """Extract plan arguments from CLI context.

    Returns
    -------
    BuildPlanArgs
        Parsed plan arguments from CLI parameters.
    """
    targets_list = ctx.params.get_list("targets")
    force_list = ctx.params.get_list("force")
    return BuildPlanArgs(
        targets=targets_list if targets_list else None,
        module=ctx.params.get_str("module"),
        force=force_list if force_list else None,
        all_targets=ctx.params.get_bool("all_targets"),
        output_file=ctx.params.get_str("output_file"),
    )


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
    execution: BuildExecutionArgs,
) -> BuildResultLike | None:
    """Execute build using Hamilton executor.

    Parameters
    ----------
    runtime
        Resolved runtime context.
    gateway
        Open storage gateway.
    execution
        BuildExecutionArgs describing goals, engine mode, force targets, and validation.

    Returns
    -------
    BuildResult | None
        Build result or None for dry-run.
    """
    if execution.run_mode is RunMode.DRY_RUN:
        LOG.info("build.cli.hamilton.dry_run goals=%s", execution.goals)
        return None

    LOG.info(
        "build.cli.hamilton.execute goals=%s force=%s mode=%s validate=%s",
        execution.goals,
        execution.force,
        execution.hamilton_mode,
        execution.validate_outputs,
    )
    providers = create_default_providers(runtime.tools)
    config = load_build_config(runtime.snapshot.repo_root)
    manifests_list = gateway.build.list_manifests(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
    )
    manifest_index = {m.target: m for m in manifests_list}
    LOG.debug("build.cli.hamilton.manifest_index count=%d", len(manifest_index))

    wrapper_allowlist_frozen = (
        frozenset(execution.wrapper_allowlist) if execution.wrapper_allowlist else None
    )

    env = BuildEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        providers=providers,
        config=config,
        profile="default",
        force_targets=frozenset(execution.force or ()),
        manifest_index=manifest_index,
        validate_outputs=execution.validate_outputs,
        strict_contracts=execution.strict_contracts,
        wrapper_allowlist=wrapper_allowlist_frozen,
    )

    executor = HamiltonBuildExecutor(profile="default", mode=execution.node_mode)
    hamilton_result = executor.run(env=env, targets=execution.goals)

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
    validate_outputs: bool
    strict_contracts: bool
    wrapper_allowlist: list[str] | None


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

    wrapper_allowlist_list = None
    wrapper_allowlist_raw = ctx.params.get_list("wrapper_allowlist")
    if wrapper_allowlist_raw:
        # Handle comma-separated string or list
        wrapper_allowlist_list = []
        for item in wrapper_allowlist_raw:
            if isinstance(item, str) and "," in item:
                wrapper_allowlist_list.extend(item.split(","))
            else:
                wrapper_allowlist_list.append(str(item))

    return _BuildRunParams(
        targets=targets,
        module=ctx.params.get_str("module"),
        all_targets=ctx.params.get_bool("all_targets"),
        dry_run=ctx.params.get_bool("dry_run"),
        force=force,
        engine=ctx.params.get_str("engine") or "hamilton",
        hamilton_mode=ctx.params.get_str("hamilton_mode") or "generated",
        validate_outputs=ctx.params.get_bool("validate_outputs"),
        strict_contracts=ctx.params.get_bool("strict_contracts"),
        wrapper_allowlist=wrapper_allowlist_list,
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

    valid_engines = ("legacy", "hamilton")
    if params.engine not in valid_engines:
        return fail_invalid_target_selection(
            f"Invalid engine '{params.engine}'. Valid: {', '.join(valid_engines)}"
        )

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

    execution_args = BuildExecutionArgs(
        goals=goals,
        force=params.force,
        run_mode=RunMode.DRY_RUN if params.dry_run else RunMode.EXECUTE,
        engine=params.engine,
        hamilton_mode=params.hamilton_mode,
        validate_outputs=params.validate_outputs,
        strict_contracts=params.strict_contracts,
        wrapper_allowlist=params.wrapper_allowlist,
    )
    return _execute_and_format_result(runtime, execution_args)


def _execute_and_format_result(
    runtime: ResolvedRuntime,
    execution: BuildExecutionArgs,
) -> CliResult[BuildRunResult]:
    """Execute build and format result.

    Parameters
    ----------
    runtime
        Resolved runtime.
    execution
        BuildExecutionArgs capturing engine, mode, validation, and goal selection.

    Returns
    -------
    CliResult[BuildRunResult]
        Build result.
    """
    try:
        with runtime_gateway(runtime, read_only=False) as gateway:
            if execution.engine == "hamilton":
                result = _execute_build_hamilton(runtime, gateway, execution)
            else:
                result, _plan = _execute_build(
                    runtime,
                    gateway,
                    execution.goals,
                    execution.force,
                    execution.run_mode,
                )
    except Exception as exc:
        LOG.exception("build.run.error engine=%s", execution.engine)
        return fail_execution_failed("build", str(exc))

    if execution.run_mode is RunMode.DRY_RUN or result is None:
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

        run_targets = gateway.build.list_run_targets(record.run_id)

        return CliResult.ok(
            BuildHistoryResult(
                runs=[record.to_dict()],
                count=1,
                targets=run_targets,
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
    try:
        _ = ctx.runtime
    except ResolutionError as e:
        return fail_project_error("build", str(e))

    graph = get_target_graph()

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

    hamilton_runtime = build_driver(mode="generated")

    dag_info = get_dag_info(hamilton_runtime, goals)

    if output_format == "mermaid":
        dag_output = export_dag_mermaid(hamilton_runtime, goals)
    elif output_format == "dot":
        dag_output = export_dag_dot(hamilton_runtime, goals)
    else:
        dag_output = export_dag_json(hamilton_runtime, goals)

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
    plan_args = _parse_plan_args(ctx)

    try:
        goals = _resolve_goals(
            targets=plan_args.targets,
            module=plan_args.module,
            target_scope=TargetScope.ALL if plan_args.all_targets else TargetScope.REQUESTED,
            graph=graph,
        )
    except ValidationError as e:
        return fail_invalid_target_selection(str(e))

    LOG.info(
        "build.plan repo=%s commit=%s targets=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        goals,
        plan_args.force,
    )

    with runtime_gateway(runtime, read_only=True) as gateway:
        providers = create_default_providers(runtime.tools)
        config = load_build_config(runtime.snapshot.repo_root)
        manifest_index = {
            m.target: m
            for m in gateway.build.list_manifests(
                repo=runtime.snapshot.repo,
                commit=runtime.snapshot.commit,
            )
        }

        env = BuildEnv(
            gateway=gateway,
            snapshot=runtime.snapshot,
            paths=runtime.paths,
            providers=providers,
            config=config,
            profile="default",
            force_targets=frozenset(plan_args.force or ()),
            manifest_index=manifest_index,
        )

        plan = compute_plan(
            env=env,
            graph=graph,
            requested=tuple(goals),
            mode="generated",
        )

    result = BuildPlanResult(
        requested=list(plan.requested),
        closure=list(plan.closure),
        entries=[e.to_dict() for e in plan.entries],
        to_compute=list(plan.to_compute),
        to_skip=list(plan.to_skip),
        blocked=list(plan.blocked),
    )

    if plan_args.output_file:
        Path(plan_args.output_file).write_text(
            _json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        LOG.info("build.plan.written path=%s", plan_args.output_file)

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

    target = ctx.params.get_str("target")
    if not target:
        return fail_invalid_targets("No target specified")

    force_list = ctx.params.get_list("force")
    force: list[str] | None = force_list if force_list else None

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
        providers = create_default_providers(runtime.tools)
        config = load_build_config(runtime.snapshot.repo_root)

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

        plan = compute_plan(
            env=env,
            graph=graph,
            requested=(target,),
            mode="generated",
        )

    entry = plan.get_entry(target)
    if entry is None:
        return fail_invalid_targets(f"Target not found in plan: {target}")

    explanation = entry.explain_staleness()

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


def build_assets_handler(ctx: CommandContext) -> CliResult[BuildAssetsResult]:
    """Handle build assets command.

    Lists materialized assets for the current snapshot with optional filters.

    Parameters
    ----------
    ctx
        Command context with gateway and runtime access.

    Returns
    -------
    CliResult[BuildAssetsResult]
        Result containing asset records and count.
    """
    gateway = ctx.gateway
    runtime = ctx.runtime
    output_format = ctx.params.get_str("output_format") or "table"

    if bool(ctx.params.get_bool("versions")):
        return _build_assets_versions_result(
            gateway=gateway, runtime=runtime, output_format=output_format, ctx=ctx
        )

    return _build_assets_legacy_result(
        gateway=gateway, runtime=runtime, output_format=output_format, ctx=ctx
    )


def _infer_asset_kind(asset_key: str) -> str:
    return "table" if "." in asset_key else "artifact"


def _looks_like_hash(value: str) -> bool:
    if len(value) not in {16, 32, 40, 64}:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def _build_assets_legacy_result(
    *,
    gateway: StorageGateway,
    runtime: ResolvedRuntime,
    output_format: str,
    ctx: CommandContext,
) -> CliResult[BuildAssetsResult]:
    asset = ctx.params.get_str("asset")
    target = ctx.params.get_str("target")
    asset_type = ctx.params.get_str("asset_type")

    assets = gateway.assets.list_assets(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
        asset_type=asset_type,
        owner_target=target,
    )
    if asset is not None:
        assets = [a for a in assets if a.asset_key == asset]

    asset_dicts = [
        {
            "asset_key": a.asset_key,
            "asset_type": a.asset_type,
            "repo": a.repo,
            "commit": a.commit,
            "owner_target": a.owner_target,
            "schema_version": a.schema_version,
            "row_count": a.row_count,
            "file_size_bytes": a.file_size_bytes,
            "materialized_at": a.materialized_at.isoformat() if a.materialized_at else None,
            "input_hash": a.input_hash,
            "metadata": a.metadata,
        }
        for a in assets
    ]
    return CliResult.ok(
        BuildAssetsResult(assets=asset_dicts, count=len(asset_dicts), format=output_format)
    )


def _build_assets_versions_result(
    *,
    gateway: StorageGateway,
    runtime: ResolvedRuntime,
    output_format: str,
    ctx: CommandContext,
) -> CliResult[BuildAssetsResult]:
    asset = ctx.params.get_str("asset")
    target = ctx.params.get_str("target")
    asset_type = ctx.params.get_str("asset_type")

    repo = runtime.snapshot.repo
    commit = runtime.snapshot.commit

    if asset is None:
        rows = gateway.execute(
            """
            SELECT DISTINCT asset_kind, asset_key
            FROM build.asset_versions
            WHERE repo = ? AND commit = ?
            ORDER BY asset_kind, asset_key
            """,
            [repo, commit],
        ).fetchall()
        assets_to_show = [(str(r[0]), str(r[1])) for r in rows]
    else:
        assets_to_show = [(_infer_asset_kind(asset), asset)]

    if target is not None:
        rows = gateway.execute(
            """
            SELECT DISTINCT asset_kind, asset_key
            FROM build.asset_versions
            WHERE repo = ? AND commit = ? AND target = ?
            ORDER BY asset_kind, asset_key
            """,
            [repo, commit, target],
        ).fetchall()
        allowed = {(str(r[0]), str(r[1])) for r in rows}
        assets_to_show = [pair for pair in assets_to_show if pair in allowed]

    if asset_type is not None:
        allowed_kinds = {"table": "table", "view": "view", "artifact": "artifact"}
        kind = allowed_kinds.get(asset_type)
        if kind is None:
            return fail_invalid_target_selection(f"Unknown asset type: {asset_type}")
        assets_to_show = [(k, a) for k, a in assets_to_show if k == kind]

    payload: list[dict[str, object]] = []
    for asset_kind, asset_key in assets_to_show:
        versions = gateway.assets.get_asset_versions(
            repo=repo,
            commit=commit,
            asset_kind=asset_kind,
            asset_key=asset_key,
            limit=50,
        )
        payload.append(
            {
                "asset_kind": asset_kind,
                "asset_key": asset_key,
                "versions": [
                    {
                        "version_hash": v.version_hash,
                        "status": v.status,
                        "run_id": v.run_id,
                        "target": v.target,
                        "impl_kind": v.impl_kind,
                        "location": v.location,
                        "input_hash": v.input_hash,
                        "options_hash": v.options_hash,
                        "schema_hash": v.schema_hash,
                        "row_count": v.row_count,
                        "bytes": v.bytes,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                        "meta": v.meta,
                    }
                    for v in versions
                ],
            }
        )

    return CliResult.ok(BuildAssetsResult(assets=payload, count=len(payload), format=output_format))


def build_lineage_handler(ctx: CommandContext) -> CliResult[BuildLineageResult]:
    """Handle build lineage command.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[BuildLineageResult]
        Lineage traversal result.
    """
    gateway = ctx.gateway
    runtime = ctx.runtime

    asset = ctx.params.get_str("asset")
    direction = (ctx.params.get_str("direction") or "up").lower()
    depth = ctx.params.get_int("depth", 1)
    output_format = ctx.params.get_str("output_format") or "json"

    if not asset:
        return fail_invalid_target_selection("Missing --asset")
    if direction not in {"up", "down"}:
        return fail_invalid_target_selection("direction must be 'up' or 'down'")
    if depth < 0:
        return fail_invalid_target_selection("depth must be >= 0")

    asset_kind = _infer_asset_kind(asset)
    root_version = gateway.assets.get_latest_version_hash(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
        asset_kind=asset_kind,
        asset_key=asset,
    )
    if root_version is None:
        return fail_invalid_targets(f"No versions recorded for asset: {asset}")

    start = (asset_kind, asset, root_version)
    node_list, edge_list = _traverse_lineage(
        gateway=gateway,
        start=start,
        direction=direction,
        depth=depth,
    )

    return CliResult.ok(
        BuildLineageResult(
            asset=asset,
            asset_kind=asset_kind,
            root_version_hash=root_version,
            direction=direction,
            depth=depth,
            nodes=node_list,
            edges=edge_list,
            format=output_format,
        )
    )


def _traverse_lineage(
    *,
    gateway: StorageGateway,
    start: tuple[str, str, str],
    direction: str,
    depth: int,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    nodes: dict[tuple[str, str, str], dict[str, str]] = {
        start: {"asset_kind": start[0], "asset_key": start[1], "version_hash": start[2]}
    }
    edges: set[tuple[tuple[str, str, str], tuple[str, str, str], str]] = set()
    frontier: set[tuple[str, str, str]] = {start}

    for _ in range(depth):
        if not frontier:
            break
        frontier = _expand_frontier(
            gateway=gateway, nodes=nodes, edges=edges, frontier=frontier, direction=direction
        )

    node_list = [nodes[k] for k in sorted(nodes)]
    edge_list: list[dict[str, object]] = []
    for a, b, edge_kind in sorted(edges):
        edge_list.append(
            {
                "from": {"asset_kind": a[0], "asset_key": a[1], "version_hash": a[2]},
                "to": {"asset_kind": b[0], "asset_key": b[1], "version_hash": b[2]},
                "edge_kind": edge_kind,
            }
        )
    return node_list, edge_list


def _expand_frontier(
    *,
    gateway: StorageGateway,
    nodes: dict[tuple[str, str, str], dict[str, str]],
    edges: set[tuple[tuple[str, str, str], tuple[str, str, str], str]],
    frontier: set[tuple[str, str, str]],
    direction: str,
) -> set[tuple[str, str, str]]:
    next_frontier: set[tuple[str, str, str]] = set()
    for kind, key, version_hash in sorted(frontier):
        if direction == "up":
            rows = gateway.execute(
                """
                SELECT upstream_kind, upstream_key, upstream_version, edge_kind
                FROM build.asset_lineage
                WHERE downstream_kind = ? AND downstream_key = ? AND downstream_version = ?
                ORDER BY upstream_kind, upstream_key, upstream_version, edge_kind
                """,
                [kind, key, version_hash],
            ).fetchall()
            for r in rows:
                upstream = (str(r[0]), str(r[1]), str(r[2]))
                edge_kind = str(r[3])
                nodes.setdefault(
                    upstream,
                    {
                        "asset_kind": upstream[0],
                        "asset_key": upstream[1],
                        "version_hash": upstream[2],
                    },
                )
                edges.add(((kind, key, version_hash), upstream, edge_kind))
                next_frontier.add(upstream)
            continue

        rows = gateway.execute(
            """
            SELECT downstream_kind, downstream_key, downstream_version, edge_kind
            FROM build.asset_lineage
            WHERE upstream_kind = ? AND upstream_key = ? AND upstream_version = ?
            ORDER BY downstream_kind, downstream_key, downstream_version, edge_kind
            """,
            [kind, key, version_hash],
        ).fetchall()
        for r in rows:
            downstream = (str(r[0]), str(r[1]), str(r[2]))
            edge_kind = str(r[3])
            nodes.setdefault(
                downstream,
                {
                    "asset_kind": downstream[0],
                    "asset_key": downstream[1],
                    "version_hash": downstream[2],
                },
            )
            edges.add((downstream, (kind, key, version_hash), edge_kind))
            next_frontier.add(downstream)

    return next_frontier


def build_promote_handler(ctx: CommandContext) -> CliResult[BuildPromoteResult]:
    """Handle build promote command.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[BuildPromoteResult]
        Promotion result.
    """
    gateway = ctx.gateway
    _ = ctx.runtime

    asset = ctx.params.get_str("asset")
    alias = ctx.params.get_str("alias")
    version_hash = ctx.params.get_str("version_hash")
    from_run_id = ctx.params.get_str("from_run_id")
    note = ctx.params.get_str("note")
    output_format = ctx.params.get_str("output_format") or "json"

    if not asset or not alias:
        return fail_invalid_target_selection("Missing --asset or --alias")

    asset_kind = _infer_asset_kind(asset)

    resolved_hash: str | None = version_hash
    if resolved_hash is None and from_run_id is not None:
        mappings = gateway.assets.get_run_asset_versions(run_id=from_run_id)
        for m in mappings:
            if m.asset_kind == asset_kind and m.asset_key == asset:
                resolved_hash = m.version_hash
                break

    if resolved_hash is None:
        return fail_invalid_target_selection("Provide --version-hash or --from-run-id")

    gateway.assets.set_alias(
        AssetAliasRecord(
            alias=alias,
            asset_kind=asset_kind,
            asset_key=asset,
            version_hash=resolved_hash,
            set_by_run_id=from_run_id,
            note=note,
        )
    )

    return CliResult.ok(
        BuildPromoteResult(
            asset=asset,
            asset_kind=asset_kind,
            alias=alias,
            version_hash=resolved_hash,
            note=note,
            format=output_format,
        )
    )


def build_resolve_handler(ctx: CommandContext) -> CliResult[BuildResolveResult]:
    """Handle build resolve command.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[BuildResolveResult]
        Alias resolution result.
    """
    gateway = ctx.gateway
    _ = ctx.runtime

    asset = ctx.params.get_str("asset")
    alias = ctx.params.get_str("alias")
    output_format = ctx.params.get_str("output_format") or "json"

    if not asset or not alias:
        return fail_invalid_target_selection("Missing --asset or --alias")

    asset_kind = _infer_asset_kind(asset)
    version_hash = gateway.assets.resolve_alias(alias=alias, asset_kind=asset_kind, asset_key=asset)
    if version_hash is None:
        return fail_invalid_targets(f"Alias not found: {alias} for {asset}")

    return CliResult.ok(
        BuildResolveResult(
            asset=asset,
            asset_kind=asset_kind,
            alias=alias,
            version_hash=version_hash,
            format=output_format,
        )
    )


def _resolve_version_spec(
    gateway: StorageGateway,
    *,
    spec: str,
    ctx: AssetVersionResolutionContext,
) -> str | None:
    if _looks_like_hash(spec):
        return spec

    resolved = gateway.assets.resolve_alias(
        alias=spec, asset_kind=ctx.asset_kind, asset_key=ctx.asset_key
    )
    if resolved is not None:
        return resolved

    if spec == "latest":
        return gateway.assets.get_latest_version_hash(
            repo=ctx.repo,
            commit=ctx.commit,
            asset_kind=ctx.asset_kind,
            asset_key=ctx.asset_key,
        )
    return None


@dataclass(frozen=True)
class AssetVersionResolutionContext:
    repo: str
    commit: str
    asset_kind: str
    asset_key: str


@dataclass(frozen=True)
class _AssetVersionRow:
    schema_hash: str | None
    row_count: int | None
    bytes: int | None


SCHEMA_ROWCOUNT_DIFF_KIND = "schema_rowcount"


def _load_asset_version_row(
    gateway: StorageGateway,
    ctx: AssetVersionResolutionContext,
    version_hash: str,
) -> _AssetVersionRow | None:
    row = gateway.execute(
        """
        SELECT schema_hash, row_count, bytes
        FROM build.asset_versions
        WHERE repo = ? AND commit = ? AND asset_kind = ? AND asset_key = ? AND version_hash = ?
        """,
        [ctx.repo, ctx.commit, ctx.asset_kind, ctx.asset_key, version_hash],
    ).fetchone()
    if row is None:
        return None
    return _AssetVersionRow(
        schema_hash=str(row[0]) if row[0] else None,
        row_count=int(row[1]) if row[1] is not None else None,
        bytes=int(row[2]) if row[2] is not None else None,
    )


def _compute_schema_rowcount_diff_summary(
    from_row: _AssetVersionRow,
    to_row: _AssetVersionRow,
) -> dict[str, object]:
    row_count_delta: int | None = None
    if isinstance(from_row.row_count, int) and isinstance(to_row.row_count, int):
        row_count_delta = to_row.row_count - from_row.row_count

    bytes_delta: int | None = None
    if isinstance(from_row.bytes, int) and isinstance(to_row.bytes, int):
        bytes_delta = to_row.bytes - from_row.bytes

    return {
        "schema": {"from": from_row.schema_hash, "to": to_row.schema_hash},
        "row_count": {"from": from_row.row_count, "to": to_row.row_count, "delta": row_count_delta},
        "bytes": {"from": from_row.bytes, "to": to_row.bytes, "delta": bytes_delta},
    }


def _get_or_compute_schema_rowcount_diff(
    gateway: StorageGateway,
    *,
    version_ctx: AssetVersionResolutionContext,
    from_hash: str,
    to_hash: str,
) -> tuple[dict[str, Any] | None, bool]:
    cached_record = gateway.assets.get_cached_diff(
        asset_kind=version_ctx.asset_kind,
        asset_key=version_ctx.asset_key,
        from_version_hash=from_hash,
        to_version_hash=to_hash,
        diff_kind=SCHEMA_ROWCOUNT_DIFF_KIND,
    )
    if cached_record is not None and cached_record.summary is not None:
        return cached_record.summary, True

    from_row = _load_asset_version_row(gateway, version_ctx, from_hash)
    to_row = _load_asset_version_row(gateway, version_ctx, to_hash)
    if from_row is None or to_row is None:
        return None, False

    summary = _compute_schema_rowcount_diff_summary(from_row, to_row)
    gateway.assets.save_cached_diff(
        AssetDiffRecord(
            asset_kind=version_ctx.asset_kind,
            asset_key=version_ctx.asset_key,
            from_version_hash=from_hash,
            to_version_hash=to_hash,
            diff_kind=SCHEMA_ROWCOUNT_DIFF_KIND,
            summary=summary,
            computed_by_run_id=None,
        )
    )
    return cast("dict[str, Any]", summary), False


def build_diff_handler(ctx: CommandContext) -> CliResult[BuildDiffResult]:
    """Handle build diff command.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[BuildDiffResult]
        Diff result.
    """
    gateway = ctx.gateway

    asset = ctx.params.get_str("asset")
    from_spec = ctx.params.get_str("from_spec")
    to_spec = ctx.params.get_str("to_spec")
    output_format = ctx.params.get_str("output_format") or "json"

    if not asset or not from_spec or not to_spec:
        return fail_invalid_target_selection("Missing --asset, --from, or --to")

    asset_kind = _infer_asset_kind(asset)
    version_ctx = AssetVersionResolutionContext(
        repo=ctx.runtime.snapshot.repo,
        commit=ctx.runtime.snapshot.commit,
        asset_kind=asset_kind,
        asset_key=asset,
    )

    from_hash = _resolve_version_spec(
        gateway,
        spec=from_spec,
        ctx=version_ctx,
    )
    to_hash = _resolve_version_spec(
        gateway,
        spec=to_spec,
        ctx=version_ctx,
    )
    if from_hash is None or to_hash is None:
        return fail_invalid_targets("Unable to resolve from/to version specs")

    diffs, cached = _get_or_compute_schema_rowcount_diff(
        gateway,
        version_ctx=version_ctx,
        from_hash=from_hash,
        to_hash=to_hash,
    )
    if diffs is None:
        return fail_invalid_targets("Version hashes not found in current snapshot catalog")

    return CliResult.ok(
        BuildDiffResult(
            asset=asset,
            asset_kind=asset_kind,
            from_spec=from_spec,
            to_spec=to_spec,
            from_version_hash=from_hash,
            to_version_hash=to_hash,
            diffs=diffs,
            cached=cached,
            format=output_format,
        )
    )


__all__ = [
    "BuildGraphResult",
    "BuildHistoryResult",
    "BuildPlanResult",
    "BuildRunResult",
    "BuildStatusResult",
    "RunMode",
    "TargetScope",
    "build_assets_handler",
    "build_diff_handler",
    "build_explain_handler",
    "build_graph_handler",
    "build_history_handler",
    "build_lineage_handler",
    "build_plan_handler",
    "build_promote_handler",
    "build_resolve_handler",
    "build_run_handler",
    "build_status_handler",
]
