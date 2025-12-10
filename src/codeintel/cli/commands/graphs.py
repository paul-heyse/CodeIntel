"""Cyclopts wiring for graph commands.

Graph analytics plugin commands using the @cli_command decorator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFmt, Verbose
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.graphs import (
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)
from codeintel.cli.rendering.types import OutputFormat
from codeintel.graphs.core.registry import DependencyPolicy, SelectionPolicy

graphs_app = App(
    name="graph",
    help="Graph analytics plugin commands.",
)

# Configuration for graph commands - no runtime/gateway needed
_GRAPH_CONFIG = CommandConfig(
    require_runtime=False,
    require_gateway=False,
)


@graphs_app.command(name="plugins-list")
@cli_command("graph.plugins.list", handler=graph_plugins_list_handler, config=_GRAPH_CONFIG)
@dataclass
class GraphPluginsListCommand:
    """List registered graph plugins."""

    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit plugin names to filter (repeatable).",
        ),
    ] = None
    include_disabled: Annotated[
        bool,
        Parameter(
            name="--include-disabled",
            help="Include disabled plugins in the listing.",
            negative=("--exclude-disabled",),
        ),
    ] = True
    output_format: OutputFmt = OutputFormat.TEXT
    verbose: Verbose = 0


@graphs_app.command(name="plugins-plan")
@cli_command("graph.plugins.plan", handler=graph_plugins_plan_handler, config=_GRAPH_CONFIG)
@dataclass
class GraphPluginsPlanCommand:
    """Display an execution plan for graph plugins."""

    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit plugin names to plan (repeatable).",
        ),
    ] = None
    enable: Annotated[
        list[str] | None,
        Parameter(
            name="--enable",
            help="Ordered list of plugins to enable (overrides defaults).",
        ),
    ] = None
    disable: Annotated[
        list[str] | None,
        Parameter(
            name="--disable",
            help="Plugins to disable/filter out from the selected set.",
        ),
    ] = None
    selection_policy: Annotated[
        SelectionPolicy,
        Parameter(
            name="--selection-policy",
            help="How to handle unknown requested plugins.",
            show_default=True,
        ),
    ] = SelectionPolicy.LENIENT
    dependency_policy: Annotated[
        DependencyPolicy,
        Parameter(
            name="--dependency-policy",
            help="How to handle missing/disabled dependencies.",
            show_default=True,
        ),
    ] = DependencyPolicy.STRICT
    output_format: OutputFmt = OutputFormat.TEXT
    verbose: Verbose = 0


__all__ = [
    "GraphPluginsListCommand",
    "GraphPluginsPlanCommand",
    "graphs_app",
]
