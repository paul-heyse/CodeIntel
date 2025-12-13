"""Graph analytics target commands.

Provide commands for listing graph build targets and their execution plans
using the Command[T] pattern. These commands now use the build registry
instead of the legacy graph plugin registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Parameter

from codeintel.build.registry import get_target_graph
from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.results import result_type

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

graphs_app = App(
    name="graph",
    help="Graph analytics target commands.",
)


def _get_graph_targets() -> list[tuple[str, str, tuple[str, ...]]]:
    """Get all targets in the graphs module.

    Returns
    -------
    list[tuple[str, str, tuple[str, ...]]]
        List of (name, description, dependencies) for each graph target.
    """
    graph = get_target_graph()
    return [
        (t.name, t.description or f"Graph target: {t.name}", t.dependencies)
        for t in graph.all_targets
        if t.module == "graphs"
    ]


@result_type
@dataclass(frozen=True)
class GraphTargetInfo:
    """Information about a single graph build target.

    Parameters
    ----------
    name
        Target name.
    description
        Target description.
    dependencies
        Target dependencies.
    tables
        Tables this target produces.
    """

    name: str
    description: str
    dependencies: list[str]
    tables: list[str]


@result_type
@dataclass(frozen=True)
class GraphTargetsResult:
    """Result from listing graph build targets.

    Parameters
    ----------
    targets
        List of target information.
    count
        Total count of targets.
    """

    targets: list[GraphTargetInfo]
    count: int


@result_type
@dataclass(frozen=True)
class GraphPlanStage:
    """A stage in the graph execution plan.

    Parameters
    ----------
    stage
        Stage number.
    targets
        Targets to execute in this stage.
    """

    stage: int
    targets: list[str]


@result_type
@dataclass(frozen=True)
class GraphPlanResult:
    """Result from planning graph target execution.

    Parameters
    ----------
    stages
        List of execution stages in dependency order.
    total_targets
        Total number of targets.
    """

    stages: list[GraphPlanStage]
    total_targets: int


@cli_command("graph.targets.list", require_storage=False)
@graphs_app.command(name="targets-list")
@dataclass(frozen=True)
class GraphTargetsList(Command[GraphTargetsResult]):
    """List graph build targets."""

    __operation_id__ = "graph.targets.list"

    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit target names to filter (repeatable).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphTargetsResult]:
        """Execute graph targets listing.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[GraphTargetsResult]
            List of targets.
        """
        _ = ctx
        names_set = set(self.names) if self.names else None

        LOG.info("Listing graph targets (names=%s)", names_set)

        graph = get_target_graph()
        targets = [t for t in graph.all_targets if t.module == "graphs"]

        if names_set:
            targets = [t for t in targets if t.name in names_set]

        target_infos = [
            GraphTargetInfo(
                name=t.name,
                description=t.description or f"Graph target: {t.name}",
                dependencies=list(t.dependencies),
                tables=list(t.table_keys),
            )
            for t in targets
        ]

        return CliResult.ok(
            GraphTargetsResult(
                targets=target_infos,
                count=len(target_infos),
            )
        )


@cli_command("graph.targets.plan", require_storage=False)
@graphs_app.command(name="targets-plan")
@dataclass(frozen=True)
class GraphTargetsPlan(Command[GraphPlanResult]):
    """Display an execution plan for graph targets in dependency order."""

    __operation_id__ = "graph.targets.plan"

    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit target names to plan (repeatable). Defaults to all graph targets.",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult]:
        """Execute graph targets planning.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[GraphPlanResult]
            Execution plan in topological order.
        """
        _ = ctx

        LOG.info("Planning graph targets (names=%s)", self.names)

        graph = get_target_graph()
        graph_targets = [t for t in graph.all_targets if t.module == "graphs"]

        if self.names:
            names_set = set(self.names)
            graph_targets = [t for t in graph_targets if t.name in names_set]

        target_names = [t.name for t in graph_targets]
        ordered = graph.topological_order(target_names) if target_names else ()

        stages = [
            GraphPlanStage(
                stage=1,
                targets=list(ordered),
            )
        ]

        return CliResult.ok(
            GraphPlanResult(
                stages=stages,
                total_targets=len(ordered),
            )
        )


@cli_command("graph.targets", require_storage=False)
@graphs_app.command(name="targets")
@dataclass(frozen=True)
class GraphTargets(Command[GraphPlanResult | GraphTargetsResult]):
    """List graph targets or show execution plan; use --plan for ordering."""

    __operation_id__ = "graph.targets"

    plan: Annotated[
        bool,
        Parameter(
            name="--plan",
            help="Display execution plan instead of listing.",
            negative=("--no-plan",),
        ),
    ] = False
    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit target names to filter or plan (repeatable).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult | GraphTargetsResult]:
        """List targets or show plan.

        Parameters
        ----------
        ctx
            Command context (unused).

        Returns
        -------
        CliResult[GraphPlanResult | GraphTargetsResult]
            Either the planned execution order or the target list.
        """
        _ = ctx
        names_set = set(self.names) if self.names else None

        graph = get_target_graph()
        graph_targets = [t for t in graph.all_targets if t.module == "graphs"]

        if names_set:
            graph_targets = [t for t in graph_targets if t.name in names_set]

        if self.plan:
            target_names = [t.name for t in graph_targets]
            ordered = graph.topological_order(target_names) if target_names else ()
            stages = [
                GraphPlanStage(
                    stage=1,
                    targets=list(ordered),
                )
            ]
            return CliResult.ok(
                GraphPlanResult(
                    stages=stages,
                    total_targets=len(ordered),
                )
            )

        target_infos = [
            GraphTargetInfo(
                name=t.name,
                description=t.description or f"Graph target: {t.name}",
                dependencies=list(t.dependencies),
                tables=list(t.table_keys),
            )
            for t in graph_targets
        ]
        return CliResult.ok(
            GraphTargetsResult(
                targets=target_infos,
                count=len(target_infos),
            )
        )


# Legacy command aliases for backward compatibility
@cli_command("graph.plugins.list", require_storage=False)
@graphs_app.command(name="plugins-list")
@dataclass(frozen=True)
class GraphPluginsList(Command[GraphTargetsResult]):
    """List graph targets (legacy alias for targets-list)."""

    __operation_id__ = "graph.plugins.list"

    names: Annotated[
        list[str] | None,
        Parameter(name="--names", help="Target names to filter."),
    ] = None
    include_disabled: Annotated[
        bool,
        Parameter(name="--include-disabled", help="Ignored (for backward compat)."),
    ] = True
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphTargetsResult]:
        """Execute listing (delegates to targets-list).

        Returns
        -------
        CliResult[GraphTargetsResult]
            List of graph targets.
        """
        _ = ctx
        cmd = GraphTargetsList(names=self.names, flags=self.flags)
        return cmd.execute(ctx)


@cli_command("graph.plugins.plan", require_storage=False)
@graphs_app.command(name="plugins-plan")
@dataclass(frozen=True)
class GraphPluginsPlan(Command[GraphPlanResult]):
    """Plan graph targets (legacy alias for targets-plan)."""

    __operation_id__ = "graph.plugins.plan"

    names: Annotated[
        list[str] | None,
        Parameter(name="--names", help="Target names to plan."),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult]:
        """Execute planning (delegates to targets-plan).

        Returns
        -------
        CliResult[GraphPlanResult]
            Execution plan.
        """
        _ = ctx
        cmd = GraphTargetsPlan(names=self.names, flags=self.flags)
        return cmd.execute(ctx)


@cli_command("graph.plugins", require_storage=False)
@graphs_app.command(name="plugins")
@dataclass(frozen=True)
class GraphPlugins(Command[GraphPlanResult | GraphTargetsResult]):
    """List or plan graph targets (legacy alias for targets)."""

    __operation_id__ = "graph.plugins"

    plan: Annotated[bool, Parameter(name="--plan", help="Show execution plan.")] = False
    names: Annotated[
        list[str] | None,
        Parameter(name="--names", help="Target names to filter."),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult | GraphTargetsResult]:
        """Execute listing or planning (delegates to targets).

        Returns
        -------
        CliResult[GraphPlanResult | GraphTargetsResult]
            Either the execution plan or the target list.
        """
        _ = ctx
        cmd = GraphTargets(plan=self.plan, names=self.names, flags=self.flags)
        return cmd.execute(ctx)


__all__ = [
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphPlugins",
    "GraphPluginsList",
    "GraphPluginsPlan",
    "GraphTargetInfo",
    "GraphTargets",
    "GraphTargetsList",
    "GraphTargetsPlan",
    "GraphTargetsResult",
    "graphs_app",
]
