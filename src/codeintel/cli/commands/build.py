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
from typing import TYPE_CHECKING, Annotated

import typer

from codeintel.build.executor import BuildExecutor, BuildResult
from codeintel.build.plan import BuildPlan, PlanGenerator
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import DatabaseState, StateValidator
from codeintel.build.targets import TargetGraph, TargetModule
from codeintel.cli.commands._common import (
    JsonOutputOpt,
    ProjectRootOpt,
    VerboseOpt,
    build_runtime_or_exit,
    setup_logging,
)

if TYPE_CHECKING:
    from codeintel.build.manifest import BuildRunRecord
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

DryRunOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        "-n",
        is_flag=True,
        help="Show build plan without executing",
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

AllOpt = Annotated[
    bool,
    typer.Option(
        "--all",
        "-a",
        is_flag=True,
        help="Build all targets across all modules",
    ),
]

# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_goals(
    targets: list[str] | None,
    module: str | None,
    all_targets: bool,
    graph: TargetGraph,
) -> list[str]:
    """Resolve target goals from CLI arguments.

    Parameters
    ----------
    targets
        Explicit target names from CLI arguments.
    module
        Module name to build all targets for.
    all_targets
        If True, build all targets across all modules.
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
    if all_targets:
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

    status_color = "succeeded" if result.success else "failed"
    lines.append("Build Complete")
    lines.append("=" * 50)
    lines.append(f"Status: {status_color}")
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
        if result.error_summary:
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
    dry_run: bool,
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
    dry_run
        If True, generate plan without executing.

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
    if dry_run:
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


# =============================================================================
# CLI Commands
# =============================================================================


@build_app.command("status")
def build_status(
    module: ModuleOpt = None,
    json_output: JsonOutputOpt = False,
    project_root: ProjectRootOpt = None,
    verbose: VerboseOpt = 0,
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
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(project_root)
    graph = get_target_graph()

    LOG.info(
        "build.status repo=%s commit=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
    )

    # Validate state
    validator = StateValidator(graph, runtime.gateway, runtime.snapshot)
    state = validator.validate()

    # Filter by module if specified
    if module:
        valid_modules: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics")
        if module not in valid_modules:
            typer.secho(
                f"Error: Unknown module: {module}. Valid: {', '.join(valid_modules)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        module_targets = graph.targets_for_module(module)  # type: ignore[arg-type]
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

    # Output
    if json_output:
        output = _format_status_json(state)
        typer.echo(json.dumps(output, indent=2))
    else:
        text = _format_status_text(state, runtime.snapshot.repo, runtime.snapshot.commit)
        typer.echo(text)


@build_app.command("run")
def build_run(
    targets: TargetsArg = None,
    module: ModuleOpt = None,
    all_targets: AllOpt = False,
    dry_run: DryRunOpt = False,
    force: ForceOpt = None,
    json_output: JsonOutputOpt = False,
    project_root: ProjectRootOpt = None,
    verbose: VerboseOpt = 0,
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
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(project_root)
    graph = get_target_graph()

    LOG.info(
        "build.run repo=%s commit=%s targets=%s module=%s all=%s dry_run=%s force=%s",
        runtime.snapshot.repo,
        runtime.snapshot.commit,
        targets,
        module,
        all_targets,
        dry_run,
        force,
    )

    # Resolve goals
    try:
        goals = _resolve_goals(targets, module, all_targets, graph)
    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Building targets: {', '.join(goals)}")
    if force:
        typer.echo(f"Forcing recompute of: {', '.join(force)}")
    if dry_run:
        typer.echo("(dry-run mode)")
    typer.echo("")

    # Execute build
    try:
        result, plan = _execute_build(runtime, goals, force, dry_run)
    except Exception as exc:
        LOG.exception("build.run.error")
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # Output
    if dry_run:
        if json_output:
            typer.echo(json.dumps(plan.to_dict(), indent=2))
        else:
            typer.echo(_format_plan_text(plan))
        return

    if result is None:
        # Should not happen if dry_run is False
        typer.secho("Error: No result from build execution", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if json_output:
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
    from codeintel.build.plan import format_duration

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
    from codeintel.build.plan import format_duration

    duration_str = (
        format_duration(record.duration_ms) if record.duration_ms else "?"
    )
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


@build_app.command("history")
def build_history(
    run_id: RunIdOpt = None,
    limit: LimitOpt = 10,
    project_root: ProjectRootOpt = None,
    json_output: JsonOutputOpt = False,
    verbose: VerboseOpt = 0,
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
    setup_logging(verbose)
    runtime = build_runtime_or_exit(project_root)

    if run_id:
        record = _lookup_run_by_id(runtime, run_id)
        if json_output:
            typer.echo(json.dumps(record.to_dict(), indent=2))
        else:
            typer.echo(_format_run_detail(record))
        return

    # List recent runs
    runs = runtime.gateway.build.list_runs(repo=runtime.snapshot.repo, limit=limit)

    if not runs:
        typer.echo("No build runs found.")
        return

    if json_output:
        typer.echo(json.dumps([r.to_dict() for r in runs], indent=2))
    else:
        typer.echo(f"Recent build runs (showing {len(runs)}):\n")
        for record in runs:
            typer.secho(_format_run_summary(record), fg=_get_status_color(record.status))


__all__ = [
    "build_app",
    "build_history",
    "build_run",
    "build_status",
]
