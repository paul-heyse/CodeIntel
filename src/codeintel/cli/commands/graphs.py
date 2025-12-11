"""Graph analytics plugin commands.

Provide commands for graph plugin listing and execution planning
using the Command[T] pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.results import result_type
from codeintel.cli.errors.results import fail_invalid_policy
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    SelectionPolicy,
    list_graph_plugins,
    plan_graph_plugins,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

graphs_app = App(
    name="graph",
    help="Graph analytics plugin commands.",
)


# =============================================================================
# Result Types
# =============================================================================


@result_type
@dataclass(frozen=True)
class GraphPluginInfo:
    """Information about a single graph plugin.

    Parameters
    ----------
    name
        Plugin name.
    description
        Plugin description.
    stage
        Plugin execution stage.
    enabled_by_default
        Whether enabled by default.
    depends_on
        Dependencies.
    provides
        What the plugin provides.
    """

    name: str
    description: str
    stage: str
    enabled_by_default: bool
    depends_on: list[str]
    provides: list[str]


@result_type
@dataclass(frozen=True)
class GraphPluginsResult:
    """Result from listing graph plugins.

    Parameters
    ----------
    plugins
        List of plugin information.
    count
        Total count of plugins.
    """

    plugins: list[GraphPluginInfo]
    count: int


@result_type
@dataclass(frozen=True)
class GraphPlanStage:
    """A stage in the graph execution plan.

    Parameters
    ----------
    stage
        Stage number.
    plugins
        Plugins to execute in this stage.
    """

    stage: int
    plugins: list[str]


@result_type
@dataclass(frozen=True)
class GraphPlanResult:
    """Result from planning graph execution.

    Parameters
    ----------
    stages
        List of execution stages.
    total_plugins
        Total number of plugins.
    disabled
        Plugins that were disabled.
    """

    stages: list[GraphPlanStage]
    total_plugins: int
    disabled: list[str]


# =============================================================================
# Commands
# =============================================================================


@cli_command("graph.plugins.list", require_storage=False)
@graphs_app.command(name="plugins-list")
@dataclass(frozen=True)
class GraphPluginsList(Command[GraphPluginsResult]):
    """List registered graph plugins."""

    __operation_id__ = "graph.plugins.list"

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
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPluginsResult]:
        """Execute graph plugins listing.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[GraphPluginsResult]
            List of plugins.
        """
        _ = ctx  # Not needed for plugin listing
        names = tuple(self.names) if self.names else None

        LOG.info("Listing graph plugins (names=%s)", names)

        plugins = list_graph_plugins()

        if names:
            plugins = [p for p in plugins if p.metadata.name in names]

        if not self.include_disabled:
            plugins = [p for p in plugins if p.metadata.enabled_by_default]

        plugin_infos = [
            GraphPluginInfo(
                name=p.metadata.name,
                description=p.metadata.description,
                stage=p.metadata.stage,
                enabled_by_default=p.metadata.enabled_by_default,
                depends_on=list(p.metadata.depends_on),
                provides=list(p.metadata.provides),
            )
            for p in plugins
        ]

        return CliResult.ok(
            GraphPluginsResult(
                plugins=plugin_infos,
                count=len(plugin_infos),
            )
        )


@cli_command("graph.plugins.plan", require_storage=False)
@graphs_app.command(name="plugins-plan")
@dataclass(frozen=True)
class GraphPluginsPlan(Command[GraphPlanResult]):
    """Display an execution plan for graph plugins."""

    __operation_id__ = "graph.plugins.plan"

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
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult]:
        """Execute graph plugins planning.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[GraphPlanResult]
            Execution plan.
        """
        _ = ctx  # Not needed for planning
        names = list(self.names) if self.names else None
        enable = list(self.enable) if self.enable else None
        disable = list(self.disable) if self.disable else None

        # Validate policies (they're already typed, but we may need to handle strings)
        selection_policy = self.selection_policy
        dependency_policy = self.dependency_policy

        if isinstance(selection_policy, str):
            try:
                selection_policy = SelectionPolicy(selection_policy)
            except ValueError:
                return fail_invalid_policy("selection", selection_policy)

        if isinstance(dependency_policy, str):
            try:
                dependency_policy = DependencyPolicy(dependency_policy)
            except ValueError:
                return fail_invalid_policy("dependency", dependency_policy)

        LOG.info(
            "Planning graph plugins (names=%s, enable=%s, disable=%s)",
            names,
            enable,
            disable,
        )

        options = PlanningOptions(
            dependency_policy=dependency_policy,
            selection_policy=selection_policy,
        )

        # Call with the correct API signature
        plan = plan_graph_plugins(
            plugin_names=names,
            enabled=enable,
            disabled=disable,
            plan_options=options,
        )

        # Convert plan to result format - GraphPluginPlan has `plugins` not `stages`
        # Simplified: treat all plugins as one stage
        plugin_names_list = [p.metadata.name for p in plan.plugins]
        stages = [
            GraphPlanStage(
                stage=1,
                plugins=plugin_names_list,
            )
        ]

        skipped_names = [skip.name for skip in plan.skipped_plugins]

        return CliResult.ok(
            GraphPlanResult(
                stages=stages,
                total_plugins=len(plan.plugins),
                disabled=skipped_names,
            )
        )


__all__ = [
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphPluginInfo",
    "GraphPluginsList",
    "GraphPluginsPlan",
    "GraphPluginsResult",
    "graphs_app",
]
