"""Build handlers.

Handlers for build operations, status, and history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.executor import BuildExecutor, BuildResult
from codeintel.build.plan import BuildPlan, PlanGenerator
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.build.targets import TargetGraph, TargetModule
from codeintel.cli.errors import ProblemDetail, ValidationError
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.cli.core import CliResult
from codeintel.storage.gateway import StorageConfig, open_gateway

if TYPE_CHECKING:
    from codeintel.build.manifest import BuildRunRecord
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)


class RunMode(Enum):
    """Build execution mode."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class TargetScope(Enum):
    """Scope selector for build goals."""

    REQUESTED = "requested"
    ALL = "all"


@dataclass(frozen=True)
class BuildRunResult:
    """Result from a build run."""

    executed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "executed": self.executed,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class BuildStatusResult:
    """Result from build status check."""

    targets: list[dict[str, object]]
    stale_count: int
    fresh_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets": self.targets,
            "stale_count": self.stale_count,
            "fresh_count": self.fresh_count,
        }


@dataclass(frozen=True)
class BuildHistoryResult:
    """Result from build history query."""

    runs: list[dict[str, object]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "runs": self.runs,
            "count": self.count,
        }


def _get_str_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: str | None = None,
) -> str | None:
    """Extract string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    str | None
        Parameter value or default.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    return str(value)


def _get_int_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: int = 0,
) -> int:
    """Extract integer parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    int
        Parameter value.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value))


def _get_bool_param(
    ctx: EnhancedHandlerContext,
    name: str,
    *,
    default: bool = False,
) -> bool:
    """Extract boolean parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    bool
        Parameter value.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime for backward compatibility.

    Parameters
    ----------
    runtime
        ResolvedRuntime from handler context.

    Returns
    -------
    ProjectRuntime
        Compatible ProjectRuntime instance.
    """
    gateway = open_gateway(StorageConfig.for_readonly(runtime.paths.db_path))
    return ProjectRuntime(
        root=runtime.root,
        project=runtime.project,
        cfg=runtime.config,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=gateway,
        tools=runtime.config.tools,
        serving=runtime.serving,
    )


def _build_runtime_from_ctx(ctx: EnhancedHandlerContext) -> ProjectRuntime:
    """Build ProjectRuntime from enhanced handler context.

    Parameters
    ----------
    ctx
        Enhanced handler context.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If project cannot be resolved.
    """
    project_root_raw = ctx.params.get("project_root")
    project_root = Path(str(project_root_raw)) if project_root_raw else None

    try:
        project_root_resolved = find_project_root(project_root)
        return build_project_runtime(project_root_resolved)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise ValidationError(msg) from exc


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
    runtime: ProjectRuntime,
    goals: list[str],
    force_targets: list[str] | None,
    run_mode: RunMode,
) -> tuple[BuildResult | None, BuildPlan]:
    """Execute build with the full Phase 2-5 pipeline.

    Parameters
    ----------
    runtime
        Project runtime context.
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
    validator = StateValidator(graph, runtime.gateway, runtime.snapshot)
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
    executor = BuildExecutor(
        graph=graph,
        gateway=runtime.gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        tools=runtime.tools,
    )
    result = executor.execute(plan)
    return result, plan


def _lookup_run_by_id(
    runtime: ProjectRuntime,
    run_id: str,
) -> BuildRunRecord:
    """Look up a build run by ID or prefix.

    Parameters
    ----------
    runtime
        Project runtime with gateway access.
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
    record = runtime.gateway.build.fetch_run(run_id)
    if record is not None:
        return record

    all_runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=100)
    matches = [r for r in all_runs if r.run_id.startswith(run_id)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        msg = f"Ambiguous run ID prefix '{run_id}' matches {len(matches)} runs"
        raise ValidationError(msg)

    msg = f"Run not found: {run_id}"
    raise ValidationError(msg)


def build_status_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[BuildStatusResult]:
    """Show current state of all build targets.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - module: Optional module filter (ingestion, graphs, analytics).

    Returns
    -------
    CliResult[BuildStatusResult]
        Structured result with target status information.
    """
    module = _get_str_param(ctx, "module")

    try:
        runtime = _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    graph = get_target_graph()

    LOG.info(
        "build.status repo=%s commit=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
    )

    validator = StateValidator(graph, runtime.gateway, runtime.snapshot)
    state = validator.validate()

    if module:
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:build:invalid-module",
                    title="Invalid Module",
                    detail=f"Unknown module: {module}. Valid: {', '.join(valid_modules)}",
                    status=400,
                )
            )

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


def _extract_build_run_params(ctx: EnhancedHandlerContext) -> _BuildRunParams:
    """Extract and normalize build run parameters from context.

    Parameters
    ----------
    ctx
        Handler context.

    Returns
    -------
    _BuildRunParams
        Extracted parameters.
    """
    targets_raw = ctx.params.get("targets")
    targets: list[str] | None
    if targets_raw is None:
        targets = None
    elif isinstance(targets_raw, list):
        targets = [str(t) for t in targets_raw]
    else:
        targets = [str(targets_raw)]

    force_raw = ctx.params.get("force")
    force: list[str] | None
    if force_raw is None:
        force = None
    elif isinstance(force_raw, list):
        force = [str(f) for f in force_raw]
    else:
        force = [str(force_raw)]

    return _BuildRunParams(
        targets=targets,
        module=_get_str_param(ctx, "module"),
        all_targets=_get_bool_param(ctx, "all_targets"),
        dry_run=_get_bool_param(ctx, "dry_run"),
        force=force,
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
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:build:invalid-module",
                    title="Invalid Module",
                    detail=f"Unknown module: {params.module}. Valid: {', '.join(valid_modules)}",
                    status=400,
                )
            )

    provided = [bool(params.targets), params.module is not None, params.all_targets]
    if sum(provided) != 1:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:invalid-selection",
                title="Invalid Target Selection",
                detail="Provide exactly one of targets, --module, or --all.",
                status=400,
            )
        )

    return None


def build_run_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[BuildRunResult]:
    """Build targets with automatic dependency resolution.

    Parameters
    ----------
    ctx
        Handler context with params:
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
        runtime = _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    graph = get_target_graph()
    scope = TargetScope.ALL if params.all_targets else TargetScope.REQUESTED

    try:
        goals = _resolve_goals(params.targets, params.module, scope, graph)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:invalid-targets",
                title="Invalid Targets",
                detail=str(e),
                status=400,
            )
        )

    LOG.info(
        "build.run repo=%s commit=%s targets=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        goals,
    )

    run_mode = RunMode.DRY_RUN if params.dry_run else RunMode.EXECUTE
    return _execute_and_format_result(runtime, goals, params.force, run_mode)


def _execute_and_format_result(
    runtime: ProjectRuntime,
    goals: list[str],
    force: list[str] | None,
    run_mode: RunMode,
) -> CliResult[BuildRunResult]:
    """Execute build and format result.

    Parameters
    ----------
    runtime
        Project runtime.
    goals
        Target goals.
    force
        Force recompute targets.
    run_mode
        Execution mode.

    Returns
    -------
    CliResult[BuildRunResult]
        Build result.
    """
    try:
        result, _plan = _execute_build(runtime, goals, force, run_mode)
    except Exception as exc:
        LOG.exception("build.run.error")
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:execution-failed",
                title="Build Execution Failed",
                detail=str(exc),
                status=500,
            )
        )

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
    ctx: EnhancedHandlerContext,
) -> CliResult[BuildHistoryResult]:
    """Show build run history and details.

    Parameters
    ----------
    ctx
        Handler context with params:
        - project_root: Optional project root override.
        - run_id: Optional specific run ID to show.
        - limit: Maximum number of runs to show (default 10).

    Returns
    -------
    CliResult[BuildHistoryResult]
        Structured result with build history.
    """
    run_id = _get_str_param(ctx, "run_id")
    limit = _get_int_param(ctx, "limit", 10)

    try:
        runtime = _build_runtime_from_ctx(ctx)
    except ValidationError as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:build:project-error",
                title="Project Error",
                detail=str(e),
                status=400,
            )
        )

    if run_id:
        try:
            record = _lookup_run_by_id(runtime, run_id)
        except ValidationError as e:
            return CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:build:run-not-found",
                    title="Run Not Found",
                    detail=str(e),
                    status=404,
                )
            )
        return CliResult.ok(
            BuildHistoryResult(
                runs=[record.to_dict()],
                count=1,
            )
        )

    runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=limit)

    return CliResult.ok(
        BuildHistoryResult(
            runs=[r.to_dict() for r in runs],
            count=len(runs),
        )
    )


__all__ = [
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "RunMode",
    "TargetScope",
    "build_history_handler",
    "build_run_handler",
    "build_status_handler",
]
