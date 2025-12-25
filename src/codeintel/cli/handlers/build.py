"""Build handlers.

Handlers for build operations, status, and history.
"""

from __future__ import annotations

import json as _json
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hamilton.caching.adapter import (
    CachingBehavior,
    CachingEventType,
    HamiltonCacheAdapter,
)

from codeintel.build.assets.impact import compute_impact
from codeintel.build.config import load_build_config
from codeintel.build.hamilton import HamiltonBuildExecutor
from codeintel.build.hamilton.decision_trace import (
    DECISION_TRACE_TARGET_NAME,
    default_decision_trace_path,
    read_decision_trace,
)
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.introspect import derive_target_io_surface
from codeintel.build.hamilton.observability import (
    export_dag_dot,
    export_dag_json,
    export_dag_mermaid,
    get_dag_info,
)
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.providers import create_default_providers
from codeintel.build.run_context import BuildRunContext, BuildRunContextOverrides
from codeintel.build.serving.publisher import (
    PublishServingSnapshotRequest,
    publish_serving_snapshot,
)
from codeintel.build.settings import get_build_settings
from codeintel.build.state import BuildState, StateValidationOptions, StateValidator
from codeintel.build.target_metadata import get_target_metadata_service, get_target_system
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
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.core.execution import ExecutionContext, RunKind, new_run_context
from codeintel.core.runtime.loader import load_execution_context, load_runtime_settings
from codeintel.observability.otel import flush_observability
from codeintel.observability.teardown import (
    ShutdownStatus,
    TeardownSnapshotOptions,
    TeardownTelemetry,
    collect_teardown_snapshot,
    emit_shutdown_error_event,
    emit_teardown_telemetry,
)
from codeintel.storage.tracking.asset_tracking import AssetAliasRecord, AssetDiffRecord
from codeintel.storage.validation import ContractValidationMode

if TYPE_CHECKING:
    from hamilton.caching.adapter import CachingEvent

    from codeintel.build.hamilton import HamiltonBuildResult
    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.hamilton.planner import HamiltonBuildPlan
    from codeintel.build.targets import TargetGraph, TargetModule
    from codeintel.cli.context import CommandContext
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.core.build_manifest import BuildRunRecord
    from codeintel.observability.cli import RunContext
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)
_TEARDOWN_DAEMON_ALLOWLIST: set[str] = set()
_CLI_INTERNAL_ERROR_STATUS_THRESHOLD = 500


@dataclass(slots=True)
class _BuildRunTelemetryState:
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class _BuildTeardownInputs:
    runtime: ResolvedRuntime | None
    goals: list[str]
    result: CliResult[BuildRunResult] | None
    run_id: str | None
    run_context: RunContext | None
    duration_ms: float


class RunMode(Enum):
    """Build execution mode."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class TargetScope(Enum):
    """Scope selector for build goals."""

    REQUESTED = "requested"
    ALL = "all"


_VALID_MODULES: tuple[str, ...] = ("ingestion", "graphs", "analytics", "export")
_CACHE_LOG_KEY_TUPLE_LEN: int = 2


def _build_execution_context(
    runtime: ResolvedRuntime,
    *,
    kind: RunKind = "full",
    requested_datasets: tuple[str, ...] = (),
) -> ExecutionContext:
    run_context = new_run_context(
        snapshot=runtime.snapshot,
        kind=kind,
        trigger="cli",
        requested_datasets=requested_datasets,
    )
    return load_execution_context(primitives=runtime.primitives, run=run_context)


@dataclass(frozen=True)
class BuildExecutionArgs:
    """Build execution options for Hamilton engine."""

    goals: list[str]
    force: list[str] | None
    run_mode: RunMode
    validate_outputs: bool
    strict_contracts: bool
    publish_serving_snapshot: bool
    parallel_backend: str
    max_workers: int | None
    enable_cache: bool
    cache_dir: str | None
    clear_cache: bool
    cache_report: bool
    validation_mode: ContractValidationMode

    @property
    def is_dry_run(self) -> bool:
        """Return True when run_mode is DRY_RUN."""
        return self.run_mode is RunMode.DRY_RUN


@dataclass(frozen=True)
class _BuildExecutionOutcome:
    """Result bundle for an executed build."""

    result: HamiltonBuildResult
    cache_report: dict[str, object] | None


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
        if module not in _VALID_MODULES:
            msg = f"Unknown module: {module}. Valid: {', '.join(_VALID_MODULES)}"
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
    state: BuildState,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Group targets by their status.

    Parameters
    ----------
    state
        Build state from StateValidator (unified types).

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        (current, stale, missing, blocked) lists.
    """
    current: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    blocked: list[str] = []

    for target_name, target_state in state.targets.items():
        if target_state.status == "current":
            current.append(target_name)
        elif target_state.status == "stale":
            reason = ""
            if target_state.blocking_reason:
                reason = f" ({target_state.blocking_reason})"
            stale.append(f"{target_name}{reason}")
        elif target_state.status == "missing":
            missing.append(target_name)
        elif target_state.status == "blocked":
            reason = ""
            if target_state.blocking_deps:
                reason = f" (blocked by: {', '.join(target_state.blocking_deps)})"
            blocked.append(f"{target_name}{reason}")

    return current, stale, missing, blocked


def _build_status_result(state: BuildState) -> BuildStatusResult:
    """Build status result from build state.

    Parameters
    ----------
    state
        Build state from validator (unified types).

    Returns
    -------
    BuildStatusResult
        Status result with counts.
    """
    targets: list[dict[str, object]] = []

    current_list, stale_list, missing_list, blocked_list = _group_targets_by_status(state)

    targets.extend({"name": name, "status": "fresh"} for name in current_list)
    targets.extend({"name": name, "status": "stale"} for name in stale_list)
    targets.extend({"name": name, "status": "missing"} for name in missing_list)
    targets.extend({"name": name, "status": "blocked"} for name in blocked_list)

    stale_count = len(stale_list) + len(missing_list)

    return BuildStatusResult(
        targets=targets,
        stale_count=stale_count,
        fresh_count=len(current_list),
    )


def _with_decision_trace_targets(targets: Sequence[str]) -> list[str]:
    """Ensure decision_trace is included in executed targets when available.

    Returns
    -------
    list[str]
        Target list including decision_trace when requested targets are present.
    """
    requested = list(targets)
    if requested and DECISION_TRACE_TARGET_NAME not in requested:
        requested.append(DECISION_TRACE_TARGET_NAME)
    return requested


def _execute_build_hamilton(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    execution: BuildExecutionArgs,
) -> _BuildExecutionOutcome | None:
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
        "build.cli.hamilton.execute goals=%s force=%s validate=%s",
        execution.goals,
        execution.force,
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

    cache_dir = runtime.paths.build_dir / ".hamilton_cache"
    if execution.cache_dir:
        override = Path(execution.cache_dir).expanduser()
        cache_dir = override if override.is_absolute() else (runtime.root / override)

    execution_context = _build_execution_context(
        runtime,
        requested_datasets=tuple(execution.goals),
    )
    execution_settings = replace(
        execution_context.execution_settings,
        parallel_backend=execution.parallel_backend,
        max_workers=execution.max_workers,
    )
    execution_options = BuildExecutionOptions(
        profile=runtime.project.default_profile,
        parallel_backend=execution.parallel_backend,
        max_workers=execution.max_workers,
        enable_hamilton_cache=execution.enable_cache,
        cache_dir=str(cache_dir),
    )
    overrides = BuildRunContextOverrides(
        run_config=None,
        execution_options=execution_options,
        force_targets=frozenset(execution.force or ()),
        validate_outputs=execution.validate_outputs,
        strict_contracts=execution.strict_contracts,
        manifest_index=manifest_index,
    )
    context = BuildRunContext.from_execution_context(
        execution_context=execution_context,
        gateway=gateway,
        providers=providers,
        config=config,
        overrides=overrides,
    )
    context = replace(context, execution_settings=execution_settings)
    env = context.build_env()

    if execution.clear_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    executor = HamiltonBuildExecutor(
        profile=execution_options.profile,
        parallel_backend=execution_options.parallel_backend,
        max_workers=execution_options.max_workers,
        enable_cache=execution_options.enable_hamilton_cache,
        cache_dir=str(cache_dir),
    )
    hamilton_result = executor.run(
        env=env,
        targets=_with_decision_trace_targets(execution.goals),
    )

    cache_report: dict[str, object] | None = None
    if execution.cache_report:
        cache_report = _build_cache_report(
            hamilton_result=hamilton_result,
            cache_dir=cache_dir,
            enable_cache=execution.enable_cache,
        )

    return _BuildExecutionOutcome(result=hamilton_result, cache_report=cache_report)


def _build_cache_report(
    *,
    hamilton_result: HamiltonBuildResult,
    cache_dir: Path,
    enable_cache: bool,
) -> dict[str, object]:
    """Build a cache hit/miss report from Hamilton cache adapter logs.

    Parameters
    ----------
    hamilton_result
        Hamilton build result for the executed run.
    cache_dir
        Cache directory used for this run.
    enable_cache
        Whether caching was enabled for this run.

    Returns
    -------
    dict[str, object]
        JSON-serializable cache report payload.
    """
    report: dict[str, object] = {
        "enabled": enable_cache,
        "cache_dir": str(cache_dir),
        "node_count": 0,
        "hit_count": 0,
        "executed_count": 0,
        "hit_rate": None,
        "nodes": [],
    }

    if not enable_cache:
        return report

    runtime = hamilton_result.runtime
    if runtime is None:
        return report

    cache_adapter = _cache_adapter_from_runtime(runtime)
    if cache_adapter is None:
        return report

    cache_run_id = _cache_last_run_id(cache_adapter)
    if cache_run_id is None:
        return report

    node_rows, hit_count, executed_count = _collect_cache_node_rows(
        runtime=runtime,
        cache_adapter=cache_adapter,
        cache_run_id=cache_run_id,
    )

    total = hit_count + executed_count
    report["node_count"] = total
    report["hit_count"] = hit_count
    report["executed_count"] = executed_count
    report["hit_rate"] = (hit_count / total) if total else None
    report["nodes"] = node_rows
    report["cache_run_id"] = cache_run_id

    return report


def _cache_adapter_from_runtime(runtime: HamiltonRuntime) -> HamiltonCacheAdapter | None:
    cache_adapter_raw = getattr(runtime.dr, "cache", None)
    if isinstance(cache_adapter_raw, HamiltonCacheAdapter):
        return cache_adapter_raw
    return None


def _cache_last_run_id(cache_adapter: HamiltonCacheAdapter) -> str | None:
    try:
        return cache_adapter.last_run_id
    except IndexError:
        return None


def _cache_log_key_parts(key: object) -> tuple[str, str | None]:
    if isinstance(key, str):
        return key, None
    if (
        isinstance(key, tuple)
        and len(key) == _CACHE_LOG_KEY_TUPLE_LEN
        and all(isinstance(x, str) for x in key)
    ):
        return key[0], key[1]
    return str(key), None


def _cache_events_outcome(events: list[CachingEvent]) -> str | None:
    if any(event.event_type == CachingEventType.GET_RESULT for event in events):
        return "hit"
    if any(event.event_type == CachingEventType.EXECUTE_NODE for event in events):
        return "executed"
    return None


def _cache_behavior_str(behavior: object) -> str | None:
    if isinstance(behavior, CachingBehavior):
        return behavior.name.lower()
    return None


def _cache_node_tag_fields(runtime: HamiltonRuntime, node_name: str) -> dict[str, object]:
    node_obj = runtime.dr.graph.nodes.get(node_name)
    if node_obj is None:
        return {}

    node_tags = node_obj.tags if isinstance(node_obj.tags, dict) else None
    if node_tags is None:
        return {}

    fields: dict[str, object] = {}
    target = node_tags.get("target")
    if isinstance(target, str):
        fields["target"] = target

    node_type = node_tags.get("node_type")
    if isinstance(node_type, str):
        fields["node_type"] = node_type

    return fields


def _collect_cache_node_rows(
    *,
    runtime: HamiltonRuntime,
    cache_adapter: HamiltonCacheAdapter,
    cache_run_id: str,
) -> tuple[list[dict[str, object]], int, int]:
    logs_by_node = cast(
        "dict[object, list[CachingEvent]]",
        cache_adapter.logs(run_id=cache_run_id, level="info"),
    )
    behavior_by_node = cache_adapter.behaviors.get(cache_run_id, {})

    node_rows: list[dict[str, object]] = []
    hit_count = 0
    executed_count = 0

    for key, events in logs_by_node.items():
        node_name, task_id = _cache_log_key_parts(key)
        outcome = _cache_events_outcome(events)
        if outcome is None:
            continue

        if outcome == "hit":
            hit_count += 1
        else:
            executed_count += 1

        row: dict[str, object] = {"node": node_name, "outcome": outcome}

        behavior_str = _cache_behavior_str(behavior_by_node.get(node_name))
        if behavior_str is not None:
            row["behavior"] = behavior_str

        if task_id is not None:
            row["task_id"] = task_id

        row.update(_cache_node_tag_fields(runtime, node_name))
        node_rows.append(row)

    node_rows.sort(key=lambda r: (str(r.get("node", "")), str(r.get("task_id", ""))))
    return node_rows, hit_count, executed_count


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
        - module: Optional module filter (ingestion, graphs, analytics, export).

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

    graph = get_target_system().graph

    LOG.info(
        "build.status repo=%s commit=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
    )

    gateway = ctx.gateway
    build_settings = get_build_settings()
    validator = StateValidator(
        graph,
        gateway,
        runtime.snapshot,
        options=StateValidationOptions(settings=build_settings),
    )
    state = validator.validate()

    if module:
        if module not in _VALID_MODULES:
            return fail_invalid_module(module, _VALID_MODULES)

        module_targets = graph.targets_for_module(cast("TargetModule", module))
        module_names = {t.name for t in module_targets}
        state = BuildState(
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
    validate_outputs: bool
    strict_contracts: bool
    publish_serving_snapshot: bool
    parallel_backend: str
    max_workers: int | None
    enable_cache: bool
    cache_dir: str | None
    clear_cache: bool
    cache_report: bool
    validation_mode: ContractValidationMode


def resolve_parallel_backend(*, parallel_backend: str | None, max_workers: int | None) -> str:
    """Resolve the effective parallel backend from CLI inputs.

    Parameters
    ----------
    parallel_backend
        Raw backend value from CLI (None means default).
    max_workers
        Requested worker count (None means not provided).

    Returns
    -------
    str
        Effective backend name.

    Notes
    -----
    ``--max-workers`` implies ``--parallel-backend=threadpool`` when the backend is left as the
    default ``sequential``. This keeps the CLI ergonomics simple while preserving a safe default.
    """
    backend = parallel_backend or "sequential"
    if max_workers is not None and backend == "sequential":
        return "threadpool"
    return backend


def _resolve_validation_mode(raw: str | None) -> ContractValidationMode:
    if raw is None:
        return ContractValidationMode.LENIENT
    normalized = raw.lower()
    try:
        return ContractValidationMode(normalized)
    except ValueError as exc:
        msg = "Invalid value for \"--validation-mode\""
        raise ValidationError(msg) from exc


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

    parallel_backend_raw = ctx.params.get_str("parallel_backend")
    max_workers_raw = ctx.params.get_int("max_workers", 0)
    max_workers = max_workers_raw or None
    parallel_backend = resolve_parallel_backend(
        parallel_backend=parallel_backend_raw,
        max_workers=max_workers,
    )

    return _BuildRunParams(
        targets=targets,
        module=ctx.params.get_str("module"),
        all_targets=ctx.params.get_bool("all_targets"),
        dry_run=ctx.params.get_bool("dry_run"),
        force=force,
        validate_outputs=ctx.params.get_bool("validate_outputs"),
        strict_contracts=ctx.params.get_bool("strict_contracts"),
        publish_serving_snapshot=ctx.params.get_bool("publish_serving_snapshot"),
        parallel_backend=parallel_backend,
        max_workers=max_workers,
        enable_cache=ctx.params.get_bool("enable_cache"),
        cache_dir=ctx.params.get_str("cache_dir"),
        clear_cache=ctx.params.get_bool("clear_cache"),
        cache_report=ctx.params.get_bool("cache_report"),
        validation_mode=_resolve_validation_mode(ctx.params.get_str("validation_mode")),
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
    error: CliResult[BuildRunResult] | None = None

    if params.module and params.module not in _VALID_MODULES:
        error = fail_invalid_module(params.module, _VALID_MODULES)

    if error is None:
        provided = [bool(params.targets), params.module is not None, params.all_targets]
        if sum(provided) != 1:
            error = fail_invalid_target_selection(
                "Provide exactly one of targets, --module, or --all."
            )

    if error is None and params.publish_serving_snapshot and params.dry_run:
        error = fail_invalid_target_selection(
            "--publish-serving-snapshot is incompatible with --dry-run."
        )

    if error is None:
        valid_backends = ("sequential", "threadpool", "auto")
        if params.parallel_backend not in valid_backends:
            error = fail_invalid_target_selection(
                f"Invalid parallel_backend '{params.parallel_backend}'. "
                f"Valid: {', '.join(valid_backends)}"
            )

    if error is None and params.max_workers is not None and params.max_workers <= 0:
        error = fail_invalid_target_selection("--workers/--max-workers must be a positive integer.")

    return error


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
        - module: Module name (ingestion, graphs, analytics, export).
        - all_targets: Build all targets.
        - dry_run: Show plan without executing.
        - force: Force recompute of specific targets.

    Returns
    -------
    CliResult[BuildRunResult]
        Structured result with build execution information.
    """
    start = time.perf_counter()
    telemetry_state = _BuildRunTelemetryState()
    runtime: ResolvedRuntime | None = None
    goals: list[str] = []
    result: CliResult[BuildRunResult] | None = None
    run_context = ctx.run_context

    try:
        result, runtime, goals = _build_run_result(
            ctx,
            telemetry_state=telemetry_state,
        )
    finally:
        teardown_inputs = _BuildTeardownInputs(
            runtime=runtime,
            goals=goals,
            result=result,
            run_id=telemetry_state.run_id,
            run_context=run_context,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        _emit_build_teardown(inputs=teardown_inputs)
    if result is None:
        return fail_execution_failed("build", "Build run failed to produce a result.")
    return result


def _build_run_result(
    ctx: CommandContext,
    *,
    telemetry_state: _BuildRunTelemetryState,
) -> tuple[CliResult[BuildRunResult] | None, ResolvedRuntime | None, list[str]]:
    runtime: ResolvedRuntime | None = None
    goals: list[str] = []
    try:
        params = _extract_build_run_params(ctx)
    except ValidationError as exc:
        return fail_invalid_target_selection(str(exc)), runtime, goals

    validation_error = _validate_build_run_params(params)
    if validation_error is not None:
        return validation_error, runtime, goals

    try:
        runtime = ctx.runtime
    except ResolutionError as exc:
        return fail_project_error("build", str(exc)), runtime, goals

    graph = get_target_metadata_service().system.graph
    scope = TargetScope.ALL if params.all_targets else TargetScope.REQUESTED
    try:
        goals = _resolve_goals(params.targets, params.module, scope, graph)
    except ValidationError as exc:
        return fail_invalid_targets(str(exc)), runtime, goals

    if params.publish_serving_snapshot and "serving_artifacts" not in goals:
        goals.append("serving_artifacts")

    LOG.info(
        "build.run repo=%s commit=%s targets=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        goals,
    )

    execution_args = BuildExecutionArgs(
        goals=goals,
        force=params.force,
        run_mode=RunMode.DRY_RUN if params.dry_run else RunMode.EXECUTE,
        validate_outputs=params.validate_outputs,
        strict_contracts=params.strict_contracts,
        publish_serving_snapshot=params.publish_serving_snapshot,
        parallel_backend=params.parallel_backend,
        max_workers=params.max_workers,
        enable_cache=params.enable_cache,
        cache_dir=params.cache_dir,
        clear_cache=params.clear_cache,
        cache_report=params.cache_report,
        validation_mode=params.validation_mode,
    )
    result = _execute_and_format_result(
        runtime,
        execution_args,
        telemetry_state=telemetry_state,
    )
    return result, runtime, goals


def _emit_build_teardown(
    *,
    inputs: _BuildTeardownInputs,
) -> None:
    """Emit teardown telemetry for build.run.

    Parameters
    ----------
    inputs
        Aggregated teardown inputs for telemetry emission.
    """
    settings = load_runtime_settings().observability
    if not settings.teardown_enabled:
        return
    repo = inputs.runtime.snapshot.repo if inputs.runtime is not None else None
    commit = inputs.runtime.snapshot.commit if inputs.runtime is not None else None
    try:
        flush_result = flush_observability()
        snapshot = collect_teardown_snapshot(
            TeardownSnapshotOptions(
                task_sample_limit=settings.teardown_task_sample_limit,
                thread_sample_limit=settings.teardown_thread_sample_limit,
                subprocess_sample_limit=settings.teardown_subprocess_sample_limit,
                allowlisted_daemon_names=_TEARDOWN_DAEMON_ALLOWLIST,
                telemetry_flush_ok=flush_result.flush_ok if flush_result is not None else None,
                telemetry_flush_ms=flush_result.flush_ms if flush_result is not None else None,
            )
        )
        cli_command = _format_cli_command(inputs.run_context)
        cli_invocation_id = (
            inputs.run_context.invocation_id if inputs.run_context is not None else None
        )
        cli_is_parse_error = False if inputs.run_context is not None else None
        telemetry = TeardownTelemetry(
            component="build",
            operation="shutdown",
            run_id=inputs.run_id,
            repo=repo,
            commit=commit,
            targets=tuple(inputs.goals),
            duration_ms=inputs.duration_ms,
            cli_invocation_id=cli_invocation_id,
            cli_command=cli_command,
            cli_exit_code=_resolve_cli_exit_code(inputs.result),
            cli_is_parse_error=cli_is_parse_error,
            cli_error_type=_resolve_cli_error_type(inputs.result),
            shutdown_status=_resolve_shutdown_status(inputs.result),
            pending_tasks_count=snapshot.pending_tasks_count,
            pending_task_samples=snapshot.pending_task_samples,
            active_threads_count=snapshot.active_threads_count,
            active_thread_names=snapshot.active_thread_names,
            subprocess_count=snapshot.subprocess_count,
            subprocess_samples=snapshot.subprocess_samples,
            telemetry_flush_ok=snapshot.telemetry_flush_ok,
            telemetry_flush_ms=snapshot.telemetry_flush_ms,
        )
        emit_teardown_telemetry(telemetry, logger=LOG)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        emit_shutdown_error_event(
            span_name="build.shutdown",
            error=exc,
            logger=LOG,
            attributes={
                key: value
                for key, value in {
                    "build.run_id": inputs.run_id,
                    "build.repo": repo,
                    "build.commit": commit,
                }.items()
                if value is not None
            },
        )
        LOG.warning("Failed to emit build teardown telemetry: %s", exc)


def _resolve_shutdown_status(result: CliResult[BuildRunResult] | None) -> ShutdownStatus:
    """Resolve teardown status from a build run result.

    Parameters
    ----------
    result
        CLI result produced by build.run.

    Returns
    -------
    str
        Status label for teardown telemetry.
    """
    if result is None:
        return "unknown"
    if not result.success:
        return "failed"
    data = result.data
    if data is not None and data.failed:
        return "partial"
    return "succeeded"


def _resolve_cli_exit_code(result: CliResult[BuildRunResult] | None) -> int | None:
    """Resolve CLI exit code from a build run result.

    Parameters
    ----------
    result
        CLI result produced by build.run.

    Returns
    -------
    int | None
        Exit code when result is present, otherwise None.
    """
    if result is None:
        return None
    if result.success:
        return 0
    error = result.error
    if error is None:
        return 1
    return 2 if error.status >= _CLI_INTERNAL_ERROR_STATUS_THRESHOLD else 1


def _resolve_cli_error_type(result: CliResult[BuildRunResult] | None) -> str | None:
    """Resolve CLI error type identifier from a build run result.

    Parameters
    ----------
    result
        CLI result produced by build.run.

    Returns
    -------
    str | None
        Error type identifier when available.
    """
    if result is None or result.error is None:
        return None
    return result.error.type


def _format_cli_command(run_context: RunContext | None) -> str | None:
    """Format CLI command chain for telemetry.

    Parameters
    ----------
    run_context
        Optional CLI run context.

    Returns
    -------
    str | None
        Dot-delimited command chain string.
    """
    if run_context is None or not run_context.command_chain:
        return None
    return ".".join(run_context.command_chain)


def _execute_and_format_result(
    runtime: ResolvedRuntime,
    execution: BuildExecutionArgs,
    *,
    telemetry_state: _BuildRunTelemetryState | None = None,
) -> CliResult[BuildRunResult]:
    """Execute build and format result.

    Parameters
    ----------
    runtime
        Resolved runtime.
    execution
        BuildExecutionArgs capturing mode, validation, and goal selection.
    telemetry_state
        Optional telemetry state to populate with execution metadata.

    Returns
    -------
    CliResult[BuildRunResult]
        Build result.
    """
    try:
        with runtime_gateway(
            runtime,
            read_only=False,
            validation_mode=execution.validation_mode,
        ) as gateway:
            outcome = _execute_build_hamilton(runtime, gateway, execution)
            if (
                execution.publish_serving_snapshot
                and execution.run_mode is RunMode.EXECUTE
                and outcome is not None
                and not outcome.result.failed_targets
            ):
                _publish_serving_snapshot_from_build(
                    runtime,
                    gateway,
                    run_id=outcome.result.run_id,
                )
    except Exception as exc:
        LOG.exception("build.run.error")
        return fail_execution_failed("build", str(exc))

    if execution.run_mode is RunMode.DRY_RUN or outcome is None:
        return CliResult.ok(
            BuildRunResult(
                executed=[],
                skipped=[],
                failed=[],
                duration_seconds=0.0,
            )
        )

    if telemetry_state is not None:
        telemetry_state.run_id = outcome.result.run_id

    cache_report = outcome.cache_report

    return CliResult.ok(
        BuildRunResult(
            executed=list(outcome.result.computed_targets),
            skipped=list(outcome.result.skipped_targets),
            failed=list(outcome.result.failed_targets),
            duration_seconds=outcome.result.duration_ms / 1000.0,
            cache=cache_report,
        )
    )


def _publish_serving_snapshot_from_build(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    *,
    run_id: str,
) -> None:
    """Publish a serving snapshot for the current build DB.

    Parameters
    ----------
    runtime
        Resolved runtime for path and snapshot identity.
    gateway
        Open write-capable gateway for checkpointing the build DB.
    run_id
        Build run identifier used to name the published snapshot.

    Raises
    ------
    FileNotFoundError
        If required serving artifacts are missing.
    """
    artifacts_dir = runtime.paths.build_dir / "serving" / "artifacts"
    semantic_registry_path = artifacts_dir / "semantic_registry.json"
    schema_manifest_path = artifacts_dir / "schema_manifest.json"
    buildspec_path = artifacts_dir / "buildspec.json"

    missing = [
        str(path)
        for path in (semantic_registry_path, schema_manifest_path, buildspec_path)
        if not path.exists()
    ]

    if missing:
        joined = ", ".join(missing)
        msg = f"Missing serving artifacts (run `codeintel build run serving_artifacts`): {joined}"
        raise FileNotFoundError(msg)

    serve_dir = load_runtime_settings().serving.serve_dir
    if not serve_dir.is_absolute():
        serve_dir = (runtime.root / serve_dir).resolve()
    else:
        serve_dir = serve_dir.resolve()

    publish_serving_snapshot(
        gateway=gateway,
        request=PublishServingSnapshotRequest(
            run_id=run_id,
            serve_dir=serve_dir,
            semantic_registry_path=semantic_registry_path,
            schema_manifest_path=schema_manifest_path,
            buildspec_path=buildspec_path,
            keep_last=10,
        ),
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

    graph = get_target_metadata_service().system.graph

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

    hamilton_runtime = build_driver()

    dag_info = get_dag_info(hamilton_runtime, goals, graph_source="hamilton")

    if output_format == "mermaid":
        dag_output = export_dag_mermaid(hamilton_runtime, goals, graph_source="hamilton")
    elif output_format == "dot":
        dag_output = export_dag_dot(hamilton_runtime, goals, graph_source="hamilton")
    else:
        dag_output = export_dag_json(hamilton_runtime, goals, graph_source="hamilton")

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
        - module: Module name (ingestion, graphs, analytics, export).
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

    graph = get_target_metadata_service().system.graph
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
        execution_context = _build_execution_context(runtime, requested_datasets=tuple(goals))
        manifest_index = {
            m.target: m
            for m in gateway.build.list_manifests(
                repo=runtime.snapshot.repo,
                commit=runtime.snapshot.commit,
            )
        }

        overrides = BuildRunContextOverrides(
            run_config=None,
            execution_options=BuildExecutionOptions(profile=runtime.project.default_profile),
            force_targets=frozenset(plan_args.force or ()),
            manifest_index=manifest_index,
        )
        context = BuildRunContext.from_execution_context(
            execution_context=execution_context,
            gateway=gateway,
            providers=providers,
            config=config,
            overrides=overrides,
        )
        env = context.build_env()

        plan = compute_plan(
            env=env,
            graph=graph,
            requested=tuple(goals),
            graph_source="hamilton",
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


def build_decision_trace_handler(ctx: CommandContext) -> CliResult[object]:
    """Show or export the latest build decision trace.

    Returns
    -------
    CliResult[object]
        CLI result containing the decision trace payload or error.
    """
    try:
        runtime = ctx.runtime
    except ResolutionError as exc:
        return fail_project_error("build", str(exc))

    input_file = ctx.params.get_str("input_file")
    output_file = ctx.params.get_str("output_file")
    trace_path = (
        Path(input_file) if input_file else default_decision_trace_path(runtime.paths.build_dir)
    )

    if not trace_path.exists():
        return fail_execution_failed(
            "build",
            f"Decision trace not found: {trace_path}",
            status=404,
        )

    try:
        payload = read_decision_trace(trace_path)
    except (OSError, ValueError) as exc:
        return fail_execution_failed("build", f"Failed to read decision trace: {exc}")

    if output_file:
        Path(output_file).write_text(
            f"{_json.dumps(payload, indent=2)}\n",
            encoding="utf-8",
        )

    metadata: dict[str, object] = {"record_count": len(payload)}
    if ctx.output_format == OutputFormat.JSON:
        return CliResult.ok(payload, metadata=metadata)
    return CliResult.ok(_json.dumps(payload, indent=2), metadata=metadata)


@dataclass(frozen=True)
class _BuildExplainParams:
    target: str
    force_targets: frozenset[str]


def _resolve_build_explain_params(
    ctx: CommandContext,
    *,
    graph: TargetGraph,
) -> _BuildExplainParams | CliResult[BuildExplainResult]:
    target = ctx.params.get_str("target")
    if not target:
        return fail_invalid_targets("No target specified")

    force_list = ctx.params.get_list("force")
    force_targets = frozenset(force_list or ())

    try:
        graph.get(target)
    except KeyError:
        return fail_invalid_targets(f"Unknown target: {target}")

    return _BuildExplainParams(
        target=target,
        force_targets=force_targets,
    )


def _compute_plan_for_explain(
    *,
    runtime: ResolvedRuntime,
    graph: TargetGraph,
    target: str,
    force_targets: frozenset[str],
) -> HamiltonBuildPlan:
    with runtime_gateway(runtime, read_only=True) as gateway:
        providers = create_default_providers(runtime.tools)
        config = load_build_config(runtime.snapshot.repo_root)
        execution_context = _build_execution_context(runtime, requested_datasets=(target,))

        manifests_list = gateway.build.list_manifests(
            repo=runtime.snapshot.repo,
            commit=runtime.snapshot.commit,
        )
        manifest_index = {m.target: m for m in manifests_list}

        overrides = BuildRunContextOverrides(
            run_config=None,
            execution_options=BuildExecutionOptions(profile=runtime.project.default_profile),
            force_targets=force_targets,
            manifest_index=manifest_index,
        )
        context = BuildRunContext.from_execution_context(
            execution_context=execution_context,
            gateway=gateway,
            providers=providers,
            config=config,
            overrides=overrides,
        )
        env = context.build_env()

        return compute_plan(
            env=env,
            graph=graph,
            requested=(target,),
            graph_source="hamilton",
        )


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

    graph = get_target_metadata_service().system.graph
    params = _resolve_build_explain_params(ctx, graph=graph)
    if isinstance(params, CliResult):
        return params

    LOG.info(
        "build.explain repo=%s commit=%s target=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        params.target,
        sorted(params.force_targets),
    )

    plan = _compute_plan_for_explain(
        runtime=runtime,
        graph=graph,
        target=params.target,
        force_targets=params.force_targets,
    )

    entry = plan.get_entry(params.target)
    if entry is None:
        return fail_invalid_targets(f"Target not found in plan: {params.target}")

    explanation = entry.explain_staleness()

    io_surface: dict[str, object] | None = None
    if ctx.params.get_bool("io_surface"):
        h_runtime = build_driver()
        surface = derive_target_io_surface(
            h_runtime,
            include_targets=(params.target,),
        ).get(params.target)
        if surface is not None:
            io_surface = asdict(surface)

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
        io_surface=io_surface,
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

    return _build_assets_result(
        gateway=gateway, runtime=runtime, output_format=output_format, ctx=ctx
    )


def _infer_asset_kind(asset_key: str) -> str:
    return "table" if "." in asset_key else "artifact"


def _looks_like_hash(value: str) -> bool:
    if len(value) not in {16, 32, 40, 64}:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def _build_assets_result(
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
            FROM build.asset_version_events
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
            FROM build.asset_version_events
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
        WHERE asset_kind = ? AND asset_key = ? AND version_hash = ?
        """,
        [ctx.asset_kind, ctx.asset_key, version_hash],
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


def build_impact_handler(ctx: CommandContext) -> CliResult[BuildImpactResult]:
    """Handle build impact command.

    Analyze downstream impact of an asset change by traversing the lineage graph.

    Parameters
    ----------
    ctx
        Command context with parameters and runtime.

    Returns
    -------
    CliResult[BuildImpactResult]
        Result containing impacted assets and targets.
    """
    gateway = ctx.gateway

    asset_kind = ctx.params.get_str("asset_kind") or "table"
    asset_key = ctx.params.get_str("asset_key") or ""
    version_hash = ctx.params.get_str("version_hash")
    show_targets = ctx.params.get_bool("show_targets")
    max_depth = ctx.params.get_int("max_depth") or 10
    output_format = ctx.params.get_str("output_format") or "json"

    if not asset_key:
        return fail_invalid_target_selection("Missing --asset-key")

    result = compute_impact(
        gateway,
        asset_kind=asset_kind,
        asset_key=asset_key,
        version_hash=version_hash,
        max_depth=max_depth,
    )

    impacted_list = [
        {
            "asset_kind": asset.asset_kind,
            "asset_key": asset.asset_key,
            "version_hash": asset.version_hash,
            "target": asset.target,
            "depth": asset.depth,
        }
        for asset in result.impacted_assets
    ]

    targets_list = sorted(result.impacted_targets) if show_targets else []

    return CliResult.ok(
        BuildImpactResult(
            source_kind=result.source_kind,
            source_key=result.source_key,
            source_version=result.source_version,
            impacted_assets=impacted_list,
            impacted_targets=targets_list,
            format=output_format,
        )
    )


@dataclass(frozen=True)
class BuildImpactResult:
    """Result of build impact analysis."""

    source_kind: str
    source_key: str
    source_version: str | None
    impacted_assets: list[dict[str, Any]]
    impacted_targets: list[str]
    format: str = "json"


__all__ = [
    "BuildGraphResult",
    "BuildHistoryResult",
    "BuildImpactResult",
    "BuildPlanResult",
    "BuildRunResult",
    "BuildStatusResult",
    "RunMode",
    "TargetScope",
    "build_assets_handler",
    "build_decision_trace_handler",
    "build_diff_handler",
    "build_explain_handler",
    "build_graph_handler",
    "build_history_handler",
    "build_impact_handler",
    "build_lineage_handler",
    "build_plan_handler",
    "build_promote_handler",
    "build_resolve_handler",
    "build_run_handler",
    "build_status_handler",
]
