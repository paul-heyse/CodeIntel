"""Build system commands for minimal-work target computation.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Build commands require complex runtime/gateway/snapshot access that is not
yet fully supported by the Command[T] pattern's Deps abstraction. They use
the handler pattern for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build import (
    build_explain_handler,
    build_graph_handler,
    build_history_handler,
    build_plan_handler,
    build_run_handler,
    build_status_handler,
)

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)


_BUILD_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("build.run", handler=build_run_handler, config=_BUILD_CONFIG)
@build_app.command(name="run")
@dataclass
class BuildRunCommand:
    """Build targets with automatic dependency resolution."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to build (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Build all targets in a module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Build all targets across all modules.",
            negative=(),
        ),
    ] = False
    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run", "-n"],
            help="Show build plan without executing.",
            negative=(),
        ),
    ] = False
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Force recompute of specific targets (repeatable).",
        ),
    ] = None
    engine: Annotated[
        str,
        Parameter(
            name=["--engine", "-e"],
            help="Build engine to use: hamilton (default) or legacy.",
            show_choices=True,
        ),
    ] = "hamilton"
    hamilton_mode: Annotated[
        str,
        Parameter(
            name=["--hamilton-mode"],
            help="Hamilton node mode: generated (default) or phase0 (debug).",
            show_choices=True,
        ),
    ] = "generated"
    validate_outputs: Annotated[
        bool,
        Parameter(
            name=["--validate-outputs"],
            help="Validate produced datasets against Pandera schemas after write.",
            negative=(),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.status", handler=build_status_handler, config=_BUILD_CONFIG)
@build_app.command(name="status")
@dataclass
class BuildStatusCommand:
    """Show current state of build targets."""

    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Filter status to a specific module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.history", handler=build_history_handler, config=_BUILD_CONFIG)
@build_app.command(name="history")
@dataclass
class BuildHistoryCommand:
    """Show build run history and details."""

    run_id: Annotated[
        str | None,
        Parameter(
            name=["--run-id", "-i"],
            help="Specific run ID to show details for (prefix match supported).",
        ),
    ] = None
    limit: Annotated[
        int,
        Parameter(
            name=["--limit", "-n"],
            help="Number of recent runs to show.",
        ),
    ] = 10
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.plan", handler=build_plan_handler, config=_BUILD_CONFIG)
@build_app.command(name="plan")
@dataclass
class BuildPlanCommand:
    """Show build plan with status and reason for each target."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to plan (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Plan all targets in a module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Plan all targets across all modules.",
            negative=(),
        ),
    ] = False
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Mark specific targets as forced (repeatable).",
        ),
    ] = None
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.explain", handler=build_explain_handler, config=_BUILD_CONFIG)
@build_app.command(name="explain")
@dataclass
class BuildExplainCommand:
    """Explain why a target is stale and what dependencies changed."""

    target: Annotated[
        str,
        Parameter(
            name=None,
            help="Target name to explain (e.g., function_metrics).",
        ),
    ]
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Mark specific targets as forced (repeatable).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.graph", handler=build_graph_handler, config=_BUILD_CONFIG)
@build_app.command(name="graph")
@dataclass
class BuildGraphCommand:
    """Export Hamilton DAG for specified targets."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to show DAG for (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Show DAG for all targets in a module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Show DAG for all targets across all modules.",
            negative=(),
        ),
    ] = False
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default), mermaid, or dot.",
        ),
    ] = "json"
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = ["build_app"]
