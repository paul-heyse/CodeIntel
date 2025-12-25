"""Graph analytics target commands.

Provide commands for listing graph build targets and their execution plans
using the Command[T] pattern. These commands now use the build registry
instead of plugin-era registries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from cyclopts import App

from codeintel.build.target_catalog import target_graph_from_catalog
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.results import result_type
from codeintel.cli.options.registry import (
    GRAPH_DEPENDENCY_POLICY,
    GRAPH_NAMES,
    GRAPH_PLAN,
    GRAPH_SELECTION_POLICY,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

graphs_app = App(
    name="graph",
    help="Graph analytics target commands.",
)

GRAPH_TARGETS_LIST_PATH: CommandPath = ("graph", "targets-list")
GRAPH_TARGETS_PLAN_PATH: CommandPath = ("graph", "targets-plan")
GRAPH_PLUGINS_PATH: CommandPath = ("graph", "plugins")
GRAPH_TARGETS_PATH: CommandPath = ("graph", "targets")

_GRAPH_TARGETS_LIST_FLAGS_FIELD = shared_flags_field(GRAPH_TARGETS_LIST_PATH)
_GRAPH_TARGETS_PLAN_FLAGS_FIELD = shared_flags_field(GRAPH_TARGETS_PLAN_PATH)
_GRAPH_PLUGINS_FLAGS_FIELD = shared_flags_field(GRAPH_PLUGINS_PATH)
_GRAPH_TARGETS_FLAGS_FIELD = shared_flags_field(GRAPH_TARGETS_PATH)


def _get_graph_targets() -> list[tuple[str, str, tuple[str, ...]]]:
    """Get all targets in the graphs module.

    Returns
    -------
    list[tuple[str, str, tuple[str, ...]]]
        List of (name, description, dependencies) for each graph target.
    """
    graph = target_graph_from_catalog()
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


class SelectionPolicy(StrEnum):
    """Policy for handling unknown plugin names."""

    LENIENT = "lenient"
    STRICT = "strict"


class DependencyPolicy(StrEnum):
    """Policy for handling missing/implicit dependencies."""

    LENIENT = "lenient"
    STRICT = "strict"


@cli_command("graph.targets.list", require_storage=False)
@graphs_app.command(name="targets-list")
@dataclass(frozen=True)
class GraphTargetsList(Command[GraphTargetsResult]):
    """List graph build targets."""

    __operation_id__ = "graph.targets.list"

    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_LIST_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_LIST_FLAGS_FIELD

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

        graph = target_graph_from_catalog()
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
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_PLAN_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_PLAN_FLAGS_FIELD

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

        graph = target_graph_from_catalog()
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


@cli_command("graph.plugins", require_storage=False)
@graphs_app.command(name="plugins")
@dataclass(frozen=True)
class GraphPlugins(Command[GraphPlanResult | GraphTargetsResult]):
    """List graph plugins or show an execution plan; use --plan for ordering."""

    __operation_id__ = "graph.plugins"

    plan: Annotated[
        bool,
        option_param(GRAPH_PLAN, command_path=GRAPH_PLUGINS_PATH),
    ] = False
    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_PLUGINS_PATH),
    ] = None
    selection_policy: Annotated[
        SelectionPolicy,
        option_param(GRAPH_SELECTION_POLICY, command_path=GRAPH_PLUGINS_PATH),
    ] = SelectionPolicy.LENIENT
    dependency_policy: Annotated[
        DependencyPolicy,
        option_param(GRAPH_DEPENDENCY_POLICY, command_path=GRAPH_PLUGINS_PATH),
    ] = DependencyPolicy.STRICT
    flags: SharedFlagsProtocol = _GRAPH_PLUGINS_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[GraphPlanResult | GraphTargetsResult]:
        """List plugins or show plan.

        Parameters
        ----------
        ctx
            Command context (unused).

        Raises
        ------
        ValueError
            When ``selection_policy`` is strict and unknown plugin names are provided.

        Returns
        -------
        CliResult[GraphPlanResult | GraphTargetsResult]
            Either the planned execution order or the plugin list.
        """
        _ = ctx
        names_set = set(self.names) if self.names else None

        graph = target_graph_from_catalog()
        graph_targets = [t for t in graph.all_targets if t.module == "graphs"]

        available_names = {t.name for t in graph_targets}

        if names_set:
            unknown = sorted(names_set - available_names)
            if unknown and self.selection_policy == SelectionPolicy.STRICT:
                msg = f"Unknown graph plugins: {unknown}"
                raise ValueError(msg)
            graph_targets = [t for t in graph_targets if t.name in names_set]

        if self.plan:
            target_names = [t.name for t in graph_targets]
            ordered = graph.topological_order(target_names) if target_names else ()

            if self.dependency_policy == DependencyPolicy.LENIENT:
                ordered = tuple(name for name in ordered if name in available_names)

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
                description=t.description or f"Graph plugin: {t.name}",
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


@cli_command("graph.targets", require_storage=False)
@graphs_app.command(name="targets")
@dataclass(frozen=True)
class GraphTargets(Command[GraphPlanResult | GraphTargetsResult]):
    """List graph targets or show execution plan; use --plan for ordering."""

    __operation_id__ = "graph.targets"

    plan: Annotated[
        bool,
        option_param(GRAPH_PLAN, command_path=GRAPH_TARGETS_PATH),
    ] = False
    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_FLAGS_FIELD

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

        graph = target_graph_from_catalog()
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


__all__ = [
    "DependencyPolicy",
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphPlugins",
    "GraphTargetInfo",
    "GraphTargets",
    "GraphTargetsList",
    "GraphTargetsPlan",
    "GraphTargetsResult",
    "SelectionPolicy",
    "graphs_app",
]
