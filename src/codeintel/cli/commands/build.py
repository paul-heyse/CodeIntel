"""Build system CLI commands for minimal-work target computation.

This module provides CLI commands for the CodeIntel build system, enabling
users to build targets with automatic dependency resolution, display
current target status, and preview build plans via dry-run.

Commands
--------
- **run**: Build targets with minimal work resolution
- **status**: Show current state of all targets

Examples
--------
Build specific targets:

    codeintel build run function_metrics

Build all targets in a module:

    codeintel build run --module analytics

Build all targets across all modules:

    codeintel build run --all

Show what would be built (dry-run):

    codeintel build run function_metrics --dry-run

Show current target status:

    codeintel build status

Force rebuild of specific targets:

    codeintel build run function_metrics --force ast
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from codeintel.build.executor import BuildExecutor, BuildResult
from codeintel.build.manifest import BuildRunRecord
from codeintel.build.plan import BuildPlan, PlanGenerator, format_duration
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.build.targets import TargetGraph, TargetModule
from codeintel.cli.commands._common import (
    JsonOutputOpt,
    OutputFormat,
    ProjectRootOpt,
    RuntimeCliOptions,
    VerboseOpt,
    build_runtime_or_exit,
    setup_logging,
)
from codeintel.cli.commands._option_shim import OptionSpec, wrap_command

if TYPE_CHECKING:
    from codeintel.cli.project import ProjectRuntime

LOG = logging.getLogger(__name__)

# =============================================================================
# CLI Type Definitions
# =============================================================================

build_app = typer.Typer(
    name="build",
    help="Build system commands for minimal-work target computation.",
    no_args_is_help=True,
)

# Option types for build commands
TargetsArg = Annotated[
    list[str] | None,
    typer.Argument(
        help="Target names to build (e.g., function_metrics, call_graph)",
    ),
]

ModuleOpt = Annotated[
    str | None,
    typer.Option(
        "--module",
        "-m",
        help="Build all targets in a module (ingestion, graphs, analytics)",
    ),
]


class RunMode(Enum):
    """Build execution mode."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


RunModeFlagOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        "-n",
        help="Show build plan without executing",
        is_flag=True,
    ),
]

ForceOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--force",
        "-f",
        help="Force recompute of specific targets (repeatable)",
    ),
]


class TargetScope(Enum):
    """Scope selector for build goals."""

    REQUESTED = "requested"
    ALL = "all"


TargetScopeFlagOpt = Annotated[
    bool,
    typer.Option(
        "--all",
        "-a",
        help="Build all targets across all modules",
        is_flag=True,
    ),
]

JsonFlagOpt = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Output as JSON (alias for --output-format json).",
        is_flag=True,
    ),
]


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
    typer.BadParameter
        If no targets specified or unknown target provided.
    """
    if target_scope is TargetScope.ALL:
        return [t.name for t in graph.all_targets]

    if module:
        # Validate module name
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            msg = f"Unknown module: {module}. Valid: {', '.join(valid_modules)}"
            raise typer.BadParameter(msg)
        module_targets = graph.targets_for_module(module)  # type: ignore[arg-type]
        return [t.name for t in module_targets]

    if targets:
        # Validate all targets exist
        for target in targets:
            try:
                graph.get(target)
            except KeyError as exc:
                msg = f"Unknown target: {target}"
                raise typer.BadParameter(msg) from exc
        return list(targets)

    msg = "Specify targets, --module, or --all"
    raise typer.BadParameter(msg)


def _build_run_context(
    project_root: Path | None = ProjectRootOpt,
    verbose: int = VerboseOpt,
    output_format: OutputFormat = JsonOutputOpt,
) -> BuildRunContext:
    return BuildRunContext(
        runtime_options=RuntimeCliOptions(project_root=project_root),
        verbose=verbose,
        output_format=output_format,
    )


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


def build_status_handler(options: BuildStatusOptions) -> None:
    """Show current state of all build targets."""
    setup_logging(options.verbose)

    runtime = build_runtime_or_exit(options.runtime_options)
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
            typer.secho(
                f"Error: Unknown module: {options.module}. Valid: {', '.join(valid_modules)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        module_targets = graph.targets_for_module(options.module)  # type: ignore[arg-type]
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
        typer.echo(json.dumps(output, indent=2))
        return

    text = _format_status_text(state, runtime.snapshot.repo, runtime.snapshot.commit)
    typer.echo(text)


# =============================================================================
# CLI Commands
# =============================================================================


@build_app.command("status")
def build_status(
    module: ModuleOpt = None,
    output_format: OutputFormat = JsonOutputOpt,
    project_root: Path | None = ProjectRootOpt,
    verbose: int = VerboseOpt,
) -> None:
    """Show current state of all build targets.

    Display which targets are computed, stale, missing, or blocked
    for the current repository and commit.

    Examples
    --------
    Show all targets:

        codeintel build status

    Show targets for a specific module:

        codeintel build status --module analytics

    Output as JSON:

        codeintel build status --json

    Raises
    ------
    typer.Exit
        If module selection is invalid or state resolution fails.
    """
    options = BuildStatusOptions(
        module=module,
        runtime_options=RuntimeCliOptions(project_root=project_root),
        output_format=output_format,
        verbose=verbose,
    )
    build_status_handler(options)


def build_run_handler(
    options: BuildRunOptions,
    ctx_opts: BuildRunContext,
) -> None:
    """Build targets with automatic dependency resolution.

    Compute the minimal work needed to bring requested targets
    up-to-date, respecting dependencies and detecting stale data.

    Examples
    --------
    Build specific targets:

        codeintel build run function_metrics call_graph

    Build all analytics targets:

        codeintel build run --module analytics

    Build all targets across all modules:

        codeintel build run --all

    Preview what would be built:

        codeintel build run function_metrics --dry-run

    Force rebuild from AST:

        codeintel build run function_metrics --force ast

    Raises
    ------
    typer.Exit
        If goal resolution fails or build execution encounters an error.
    """
    setup_logging(ctx_opts.verbose)

    run_options = options
    run_ctx = ctx_opts

    runtime = build_runtime_or_exit(run_ctx.runtime_options)
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
    try:
        goals = _resolve_goals(
            run_options.targets, run_options.module, run_options.target_scope, graph
        )
    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Building targets: {', '.join(goals)}")
    if run_options.force:
        typer.echo(f"Forcing recompute of: {', '.join(run_options.force)}")
    if run_options.run_mode is RunMode.DRY_RUN:
        typer.echo("(dry-run mode)")
    typer.echo("")

    # Execute build
    try:
        result, plan = _execute_build(runtime, goals, run_options.force, run_options.run_mode)
    except Exception as exc:
        LOG.exception("build.run.error")
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # Output
    if run_options.run_mode is RunMode.DRY_RUN:
        if run_ctx.output_format is OutputFormat.JSON:
            typer.echo(json.dumps(plan.to_dict(), indent=2))
        else:
            typer.echo(_format_plan_text(plan))
        return

    if result is None:
        # Should not happen if dry_run is False
        typer.secho("Error: No result from build execution", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if run_ctx.output_format is OutputFormat.JSON:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(_format_result_text(result))

    # Exit with error if build failed
    if result.failed_targets:
        typer.secho(
            f"Build failed: {len(result.failed_targets)} targets",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho("Build completed successfully", fg=typer.colors.GREEN)


def _bundle_build_run(cli_kwargs: Mapping[str, object]) -> dict[str, object]:
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


build_run_option_specs = [
    OptionSpec("targets", TargetsArg, None),
    OptionSpec("module", ModuleOpt, None),
    OptionSpec("target_scope", TargetScopeFlagOpt, default=False),
    OptionSpec("run_mode", RunModeFlagOpt, default=False),
    OptionSpec("force", ForceOpt, None),
    OptionSpec("project_root", Path | None, ProjectRootOpt),
    OptionSpec("verbose", int, VerboseOpt),
    OptionSpec("output_format", OutputFormat, JsonOutputOpt),
    OptionSpec("json", JsonFlagOpt, default=False),
]

build_run = build_app.command("run")(
    wrap_command(
        build_run_handler,
        build_run_option_specs,
        bundle=_bundle_build_run,
        name="build_run",
    )
)


# =============================================================================
# History Command
# =============================================================================

RunIdOpt = Annotated[
    str | None,
    typer.Option(
        "--run-id",
        "-i",
        help="Specific run ID to show details for (prefix match supported)",
    ),
]

LimitOpt = Annotated[
    int,
    typer.Option(
        "--limit",
        "-n",
        help="Number of recent runs to show",
    ),
]


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
    typer.Exit
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
        typer.secho(
            f"Ambiguous run ID prefix '{run_id}' matches {len(matches)} runs:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        for r in matches[:5]:
            typer.echo(f"  {r.run_id}", err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Error: Run not found: {run_id}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _get_status_color(status: str) -> str:
    """Get color for a build status.

    Parameters
    ----------
    status
        Build status string.

    Returns
    -------
    str
        Typer color constant.
    """
    if status == "succeeded":
        return typer.colors.GREEN
    if status == "failed":
        return typer.colors.RED
    return typer.colors.YELLOW


def build_history_handler(options: BuildHistoryOptions) -> None:
    """Show build run history and details."""
    setup_logging(options.verbose)
    runtime = build_runtime_or_exit(options.runtime_options)

    if options.run_id:
        record = _lookup_run_by_id(runtime, options.run_id)
        if options.output_format is OutputFormat.JSON:
            typer.echo(json.dumps(record.to_dict(), indent=2))
            return
        typer.echo(_format_run_detail(record))
        return

    runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=options.limit)

    if not runs:
        typer.echo("No build runs found.")
        return

    if options.output_format is OutputFormat.JSON:
        typer.echo(json.dumps([r.to_dict() for r in runs], indent=2))
        return

    typer.echo(f"Recent build runs (showing {len(runs)}):\n")
    for record in runs:
        typer.secho(_format_run_summary(record), fg=_get_status_color(record.status))


@build_app.command("history")
def build_history(
    run_id: RunIdOpt = None,
    limit: LimitOpt = 10,
    project_root: Path | None = ProjectRootOpt,
    output_format: OutputFormat = JsonOutputOpt,
    verbose: int = VerboseOpt,
) -> None:
    """Show build run history and details.

    Lists recent build runs or shows details for a specific run.
    Use --run-id to see detailed information about a specific run,
    including which targets were computed and which were skipped.

    Examples
    --------
    .. code-block:: bash

        # Show recent build runs
        codeintel build history

        # Show more runs
        codeintel build history --limit 20

        # Show details for a specific run
        codeintel build history --run-id abc12345

        # Output as JSON
        codeintel build history --json
    """
    options = BuildHistoryOptions(
        run_id=run_id,
        limit=limit,
        runtime_options=RuntimeCliOptions(project_root=project_root),
        output_format=output_format,
        verbose=verbose,
    )
    build_history_handler(options)


__all__ = [
    "BuildHistoryOptions",
    "BuildRunContext",
    "BuildRunOptions",
    "BuildStatusOptions",
    "build_app",
    "build_history",
    "build_history_handler",
    "build_run",
    "build_run_handler",
    "build_status",
    "build_status_handler",
]
