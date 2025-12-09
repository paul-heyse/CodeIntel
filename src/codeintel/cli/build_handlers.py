"""Typer-free handlers for build system commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from codeintel.build.executor import BuildExecutor, BuildResult
from codeintel.build.manifest import BuildRunRecord
from codeintel.build.plan import BuildPlan, PlanGenerator, format_duration
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.build.targets import TargetGraph, TargetModule
from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.common_handlers import OutputFormat, RuntimeCliOptions
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)

LOG = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RunMode(Enum):
    """Build execution mode."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class TargetScope(Enum):
    """Scope selector for build goals."""

    REQUESTED = "requested"
    ALL = "all"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class BuildRunOptions:
    """Selection and execution options for a build run."""

    targets: list[str] | None
    module: str | None
    target_scope: TargetScope
    run_mode: RunMode
    force: list[str] | None


@dataclass(frozen=True)
class BuildRunContext:
    """Execution context options for a build run."""

    runtime_options: RuntimeCliOptions
    verbose: int
    output_format: OutputFormat


@dataclass(frozen=True)
class BuildStatusOptions:
    """Options for the build status command."""

    module: str | None
    runtime_options: RuntimeCliOptions
    output_format: OutputFormat
    verbose: int


@dataclass(frozen=True)
class BuildHistoryOptions:
    """Options for the build history command."""

    run_id: str | None
    limit: int
    runtime_options: RuntimeCliOptions
    output_format: OutputFormat
    verbose: int


# =============================================================================
# Logging Configuration
# =============================================================================


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Parameters
    ----------
    verbosity
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# =============================================================================
# Runtime Utilities
# =============================================================================


def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Build a ProjectRuntime from CLI options.

    Parameters
    ----------
    options
        CLI options containing project root.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If the project cannot be resolved.
    """
    try:
        project_root = find_project_root(options.project_root)
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        message = f"Project not found: {exc}"
        raise ValidationError(message) from exc
    except Exception as exc:
        message = f"Failed to load project: {exc}"
        raise ValidationError(message) from exc


# =============================================================================
# Helper Functions
# =============================================================================


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
        # Validate module name
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            msg = f"Unknown module: {module}. Valid: {', '.join(valid_modules)}"
            raise ValidationError(msg)
        module_typed = cast("TargetModule", module)
        module_targets = graph.targets_for_module(module_typed)
        return [t.name for t in module_targets]

    if targets:
        # Validate all targets exist
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


def _format_chunked_section(
    lines: list[str],
    header: str,
    items: list[str],
    marker: str,
    chunk_size: int = 5,
) -> None:
    """Append a chunked section to lines.

    Parameters
    ----------
    lines
        Output lines to append to.
    header
        Section header text.
    items
        Items to display.
    marker
        Prefix marker for each line.
    chunk_size
        Number of items per line.
    """
    if not items:
        return
    lines.append(f"{header} ({len(items)}):")
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        lines.append(f"  {marker} {', '.join(chunk)}")
    lines.append("")


def _format_list_section(
    lines: list[str],
    header: str,
    items: list[str],
    marker: str,
) -> None:
    """Append a list section to lines.

    Parameters
    ----------
    lines
        Output lines to append to.
    header
        Section header text.
    items
        Items to display.
    marker
        Prefix marker for each line.
    """
    if not items:
        return
    lines.append(f"{header} ({len(items)}):")
    lines.extend(f"  {marker} {item}" for item in items)
    lines.append("")


def _format_status_text(state: DatabaseState, repo: str, commit: str) -> str:
    """Format database state as human-readable text.

    Parameters
    ----------
    state
        Database state from StateValidator.
    repo
        Repository name.
    commit
        Commit SHA.

    Returns
    -------
    str
        Formatted status text.
    """
    lines: list[str] = [
        f"Database State for {repo} @ {commit[:8]}",
        "=" * 50,
        "",
    ]

    computed, stale, missing, blocked = _group_targets_by_status(state)

    _format_chunked_section(lines, "Computed", computed, "✓")
    _format_list_section(lines, "Stale", stale, "⚠")
    _format_chunked_section(lines, "Missing", missing, "✗")
    _format_list_section(lines, "Blocked", blocked, "⊘")

    return "\n".join(lines)


def _format_status_json(state: DatabaseState) -> dict[str, list[str]]:
    """Format database state as JSON-serializable dict.

    Parameters
    ----------
    state
        Database state from StateValidator.

    Returns
    -------
    dict[str, list[str]]
        Status grouped by category.
    """
    result: dict[str, list[str]] = {
        "computed": [],
        "stale": [],
        "missing": [],
        "blocked": [],
    }

    for target_name, target_state in state.targets.items():
        if target_state.status == "computed":
            result["computed"].append(target_name)
        elif target_state.status == "stale":
            result["stale"].append(target_name)
        elif target_state.status == "missing":
            result["missing"].append(target_name)
        elif target_state.status == "blocked":
            result["blocked"].append(target_name)

    return result


def _format_result_text(result: BuildResult) -> str:
    """Format build result as human-readable text.

    Includes rich error messages with actionable hints when available.

    Parameters
    ----------
    result
        Build execution result.

    Returns
    -------
    str
        Formatted result text.
    """
    lines: list[str] = []

    status_text = "succeeded" if result.success else "failed"
    lines.append("Build Complete")
    lines.append("=" * 50)
    lines.append(f"Status: {status_text}")
    lines.append(f"Run ID: {result.run_id}")
    lines.append(f"Duration: {result.duration_ms / 1000:.1f}s")
    lines.append("")

    if result.completed_targets:
        lines.append(f"Completed: {len(result.completed_targets)} targets")
        lines.append(f"  ✓ {', '.join(result.completed_targets)}")
        lines.append("")

    if result.skipped_targets:
        lines.append(f"Skipped: {len(result.skipped_targets)} targets (already current)")
        lines.append(f"  ○ {', '.join(result.skipped_targets)}")
        lines.append("")

    if result.failed_targets:
        lines.append(f"Failed: {len(result.failed_targets)} targets")
        lines.append(f"  ✗ {', '.join(result.failed_targets)}")
        lines.append("")

        # Display rich error messages with actionable hints
        if result.errors.has_errors:
            lines.append("Error Details:")
            lines.append("-" * 40)
            for i, error in enumerate(result.errors.errors, 1):
                lines.append(f"  {i}. {error.error_code}")
                lines.append(f"     {error.user_message}")
                if error.actionable_hint:
                    lines.append(f"     Hint: {error.actionable_hint}")
                lines.append("")
        elif result.error_summary:
            # Fallback to simple error summary
            lines.append(f"  Error: {result.error_summary}")
            lines.append("")

    return "\n".join(lines)


def _format_plan_text(plan: BuildPlan) -> str:
    """Format build plan as human-readable text.

    Parameters
    ----------
    plan
        Build plan from PlanGenerator.

    Returns
    -------
    str
        Formatted plan text.
    """
    return plan.format_summary()


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

    # Phase 2: Validate state
    LOG.info("build.cli.validate_state goals=%s", goals)
    validator = StateValidator(graph, runtime.gateway, runtime.snapshot)
    state = validator.validate()

    # Phase 3: Resolve minimal work
    LOG.info("build.cli.resolve force=%s", force_targets)
    resolver = BuildResolver(graph, state)
    resolution = resolver.resolve(goals, force_recompute=tuple(force_targets or ()))

    # Phase 4: Generate plan
    LOG.info("build.cli.generate_plan to_compute=%d", len(resolution.to_compute))
    generator = PlanGenerator(graph)
    plan = generator.generate(resolution)

    # Phase 5: Execute or return dry-run result
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


def _format_run_detail(record: BuildRunRecord) -> str:
    """Format a single build run for detailed display.

    Parameters
    ----------
    record
        Build run record to format.

    Returns
    -------
    str
        Formatted run details.
    """
    lines = [
        f"Run: {record.run_id}",
        f"  Repo:    {record.repo}",
        f"  Commit:  {record.commit[:12]}",
        f"  Status:  {record.status}",
        f"  Started: {record.started_at.isoformat() if record.started_at else 'N/A'}",
    ]
    if record.completed_at:
        lines.append(f"  Ended:   {record.completed_at.isoformat()}")
    if record.duration_ms is not None:
        lines.append(f"  Duration: {format_duration(record.duration_ms)}")

    if record.requested_targets:
        lines.append(f"  Requested ({len(record.requested_targets)}):")
        lines.extend(f"    - {t}" for t in sorted(record.requested_targets))

    if record.computed_targets:
        lines.append(f"  Computed ({len(record.computed_targets)}):")
        lines.extend(f"    ✓ {t}" for t in sorted(record.computed_targets))

    if record.skipped_targets:
        lines.append(f"  Skipped ({len(record.skipped_targets)}):")
        lines.extend(f"    - {t}" for t in sorted(record.skipped_targets))

    if record.error_summary:
        lines.append(f"  Error: {record.error_summary}")

    return "\n".join(lines)


def _format_run_summary(record: BuildRunRecord) -> str:
    """Format a single build run for list display.

    Parameters
    ----------
    record
        Build run record to format.

    Returns
    -------
    str
        Formatted single-line summary.
    """
    duration_str = format_duration(record.duration_ms) if record.duration_ms else "?"
    computed_count = len(record.computed_targets)
    skipped_count = len(record.skipped_targets)

    return (
        f"{record.run_id[:8]}  "
        f"{record.status:<10}  "
        f"{computed_count:>2} computed, {skipped_count:>2} skipped  "
        f"{duration_str:>8}  "
        f"{record.started_at.strftime('%Y-%m-%d %H:%M')}"
    )


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
    # First try exact match
    record = runtime.gateway.build.fetch_run(run_id)
    if record is not None:
        return record

    # Try prefix match
    all_runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=100)
    matches = [r for r in all_runs if r.run_id.startswith(run_id)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        sys.stderr.write(f"Ambiguous run ID prefix '{run_id}' matches {len(matches)} runs:\n")
        for r in matches[:5]:
            sys.stderr.write(f"  {r.run_id}\n")
        msg = f"Ambiguous run ID prefix '{run_id}'"
        raise ValidationError(msg)

    msg = f"Run not found: {run_id}"
    raise ValidationError(msg)


# =============================================================================
# Bundle Function
# =============================================================================


def bundle_build_run(cli_kwargs: Mapping[str, object]) -> dict[str, object]:
    """Bundle CLI arguments into typed options for build run.

    Parameters
    ----------
    cli_kwargs
        Raw CLI keyword arguments.

    Returns
    -------
    dict[str, object]
        Bundled options and context.
    """
    output_format = cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT))
    json_override = bool(cli_kwargs.get("json", False))
    if json_override:
        output_format = OutputFormat.JSON
    target_scope_flag = bool(cli_kwargs.get("target_scope", False))
    run_mode_flag = bool(cli_kwargs.get("run_mode", False))
    options = BuildRunOptions(
        targets=cast("list[str] | None", cli_kwargs.get("targets")),
        module=cast("str | None", cli_kwargs.get("module")),
        target_scope=TargetScope.ALL if target_scope_flag else TargetScope.REQUESTED,
        run_mode=RunMode.DRY_RUN if run_mode_flag else RunMode.EXECUTE,
        force=cast("list[str] | None", cli_kwargs.get("force")),
    )
    ctx_opts = BuildRunContext(
        runtime_options=RuntimeCliOptions(
            project_root=cast("Path | None", cli_kwargs.get("project_root")),
        ),
        verbose=int(cast("int", cli_kwargs.get("verbose", 0))),
        output_format=output_format,
    )
    return {"options": options, "ctx_opts": ctx_opts}


# =============================================================================
# Handlers
# =============================================================================


def build_status_handler(options: BuildStatusOptions) -> None:
    """Show current state of all build targets.

    Parameters
    ----------
    options
        Status command options.

    Raises
    ------
    ValidationError
        If module selection is invalid.
    """
    setup_logging(options.verbose)

    runtime = build_runtime_from_cli(options.runtime_options)
    graph = get_target_graph()

    LOG.info(
        "build.status repo=%s commit=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
    )

    validator = StateValidator(graph, runtime.gateway, runtime.snapshot)
    state = validator.validate()

    if options.module:
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if options.module not in valid_modules:
            msg = f"Unknown module: {options.module}. Valid: {', '.join(valid_modules)}"
            raise ValidationError(msg)

        module_targets = graph.targets_for_module(cast("TargetModule", options.module))
        module_names = {t.name for t in module_targets}
        filtered_targets = {
            name: target_state
            for name, target_state in state.targets.items()
            if name in module_names
        }
        state = DatabaseState(
            repo=state.repo,
            commit=state.commit,
            targets=filtered_targets,
        )

    if options.output_format is OutputFormat.JSON:
        output = _format_status_json(state)
        sys.stdout.write(json.dumps(output, indent=2))
        sys.stdout.write("\n")
        return

    text = _format_status_text(state, runtime.snapshot.repo, runtime.snapshot.commit)
    sys.stdout.write(text)
    sys.stdout.write("\n")


def build_run_handler(
    options: BuildRunOptions,
    ctx_opts: BuildRunContext,
) -> None:
    """Build targets with automatic dependency resolution.

    Compute the minimal work needed to bring requested targets
    up-to-date, respecting dependencies and detecting stale data.

    Parameters
    ----------
    options
        Build run selection options.
    ctx_opts
        Execution context options.

    Raises
    ------
    ValidationError
        If goal resolution fails or build execution encounters an error.
    """
    setup_logging(ctx_opts.verbose)

    run_options = options
    run_ctx = ctx_opts

    runtime = build_runtime_from_cli(run_ctx.runtime_options)
    graph = get_target_graph()

    LOG.info(
        "build.run repo=%s commit=%s targets=%s module=%s scope=%s run_mode=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        run_options.targets,
        run_options.module,
        run_options.target_scope,
        run_options.run_mode,
        run_options.force,
    )

    # Resolve goals
    goals = _resolve_goals(run_options.targets, run_options.module, run_options.target_scope, graph)

    sys.stdout.write(f"Building targets: {', '.join(goals)}\n")
    if run_options.force:
        sys.stdout.write(f"Forcing recompute of: {', '.join(run_options.force)}\n")
    if run_options.run_mode is RunMode.DRY_RUN:
        sys.stdout.write("(dry-run mode)\n")
    sys.stdout.write("\n")

    # Execute build
    try:
        result, plan = _execute_build(runtime, goals, run_options.force, run_options.run_mode)
    except Exception as exc:
        LOG.exception("build.run.error")
        raise ValidationError(str(exc)) from exc

    # Output
    if run_options.run_mode is RunMode.DRY_RUN:
        if run_ctx.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(plan.to_dict(), indent=2))
            sys.stdout.write("\n")
        else:
            sys.stdout.write(_format_plan_text(plan))
            sys.stdout.write("\n")
        return

    if result is None:
        # Should not happen if dry_run is False
        msg = "No result from build execution"
        raise ValidationError(msg)

    if run_ctx.output_format is OutputFormat.JSON:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_result_text(result))
        sys.stdout.write("\n")

    # Exit with error if build failed
    if result.failed_targets:
        msg = f"Build failed: {len(result.failed_targets)} targets"
        raise ValidationError(msg)

    sys.stdout.write("Build completed successfully\n")


def build_history_handler(options: BuildHistoryOptions) -> None:
    """Show build run history and details.

    Parameters
    ----------
    options
        History command options.
    """
    setup_logging(options.verbose)
    runtime = build_runtime_from_cli(options.runtime_options)

    if options.run_id:
        record = _lookup_run_by_id(runtime, options.run_id)
        if options.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(record.to_dict(), indent=2))
            sys.stdout.write("\n")
            return
        sys.stdout.write(_format_run_detail(record))
        sys.stdout.write("\n")
        return

    runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=options.limit)

    if not runs:
        sys.stdout.write("No build runs found.\n")
        return

    if options.output_format is OutputFormat.JSON:
        sys.stdout.write(json.dumps([r.to_dict() for r in runs], indent=2))
        sys.stdout.write("\n")
        return

    sys.stdout.write(f"Recent build runs (showing {len(runs)}):\n\n")
    for record in runs:
        sys.stdout.write(_format_run_summary(record))
        sys.stdout.write("\n")


__all__ = [
    "BuildHistoryOptions",
    "BuildRunContext",
    "BuildRunOptions",
    "BuildStatusOptions",
    "OutputFormat",
    "RunMode",
    "RuntimeCliOptions",
    "TargetScope",
    "build_history_handler",
    "build_run_handler",
    "build_runtime_from_cli",
    "build_status_handler",
    "bundle_build_run",
    "setup_logging",
]
