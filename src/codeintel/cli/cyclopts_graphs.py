"""Cyclopts wiring for graph commands.

This module wires Cyclopts command classes to unified handlers via command_context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.handlers.graphs import (
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)
from codeintel.graphs.core.registry import DependencyPolicy, SelectionPolicy

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
        # Build runtime and output CLI objects
        runtime_cli = RuntimeCLI(verbose=self.verbose)
        output_cli = OutputFormatCLI(output_format=self.output_format)

        # Build params dict
        params: dict[str, object] = {
            "names": tuple(self.names) if self.names else None,
            "enable": tuple(self.enable) if self.enable else None,
            "disable": tuple(self.disable) if self.disable else (),
            "selection_policy": self.selection_policy.value,
            "dependency_policy": self.dependency_policy.value,
            "include_disabled": True,  # Include all when listing
        }

        with command_context(
            "graph.plugins",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,  # No project needed for listing plugins
        ) as (ctx, renderer):
            # Choose handler based on mode and render
            if self.plan or self.validate_plan:
                plan_result = graph_plugins_plan_handler(ctx)
                exit_code = renderer.render_result(plan_result)
            else:
                list_result = graph_plugins_list_handler(ctx)
                exit_code = renderer.render_result(list_result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["graphs_app"]
