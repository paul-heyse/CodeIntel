"""Cyclopts wiring for graph commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.graphs_handlers import (
    DependencyPolicy,
    SelectionPolicy,
    graph_plugins_ctx,
)

graphs_app = App(
    name="graph",
    help="Graph analytics plugin commands.",
)


@graphs_app.command(name="plugins")
@dataclass
class GraphPluginsCommand:
    """List registered graph plugins or display an execution plan."""

    plan: Annotated[
        bool,
        Parameter(
            name="--plan",
            help="Show planned execution order instead of a simple list.",
            negative=(),
        ),
    ] = False
    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit plugin names to plan/list (repeatable).",
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
    validate_plan: Annotated[
        bool,
        Parameter(
            name="--validate-plan",
            help="Validate plan strictly (selection/dependency strict).",
            negative=(),
        ),
    ] = False
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the graph plugins command."""
        CycloptsAdapter("graph.plugins", graph_plugins_ctx)(self)


__all__ = ["graphs_app"]
