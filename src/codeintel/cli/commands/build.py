"""Cyclopts wiring for the build command group.

This module wires Cyclopts command classes to unified handlers via @cli_command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build import (
    build_history_handler,
    build_run_handler,
    build_status_handler,
)

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)

# Config for build commands - requires runtime and gateway
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = ["build_app"]
