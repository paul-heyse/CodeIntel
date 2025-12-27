"""Graph handlers.

Handlers for graph target listing and execution planning.
These handlers use the build registry instead of the legacy graph plugin registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "tables": self.tables,
        }


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets": [t.to_dict() for t in self.targets],
            "count": self.count,
        }


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "stage": self.stage,
            "targets": self.targets,
        }


@dataclass(frozen=True)
class GraphPlanResult:
    """Result from planning graph target execution.

    Parameters
    ----------
    stages
        List of execution stages.
    total_targets
        Total number of targets.
    """

    stages: list[GraphPlanStage]
    total_targets: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_targets": self.total_targets,
        }


def graph_targets_list_handler(
    ctx: CommandContext,
) -> CliResult[GraphTargetsResult]:
    """List graph build targets.

    Parameters
    ----------
    ctx
        Command context with params:
        - names: Optional target names to filter.

    Returns
    -------
    CliResult[GraphTargetsResult]
        List of targets.
    """
    names_tuple = ctx.params.get_tuple("names")
    names_set: set[str] | None = set(names_tuple) if names_tuple else None

    LOG.info("Listing graph targets (names=%s)", names_set)

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    catalog = runtime_bundle.catalog
    targets = [t for t in catalog.all_targets if t.module == "graphs"]

    if names_set:
        targets = [t for t in targets if t.name in names_set]

    target_infos = [
        GraphTargetInfo(
            name=t.name,
            description=t.description or f"Graph target: {t.name}",
            dependencies=list(t.dependencies),
            tables=[output.key for output in catalog.table_outputs_by_target.get(t.name, ())],
        )
        for t in targets
    ]

    return CliResult.ok(
        GraphTargetsResult(
            targets=target_infos,
            count=len(target_infos),
        )
    )


def graph_targets_plan_handler(
    ctx: CommandContext,
) -> CliResult[GraphPlanResult]:
    """Display an execution plan for graph targets in dependency order.

    Parameters
    ----------
    ctx
        Command context with params:
        - names: Optional target names to include.

    Returns
    -------
    CliResult[GraphPlanResult]
        Execution plan in topological order.
    """
    names_tuple = ctx.params.get_tuple("names")
    names_set: set[str] | None = set(names_tuple) if names_tuple else None

    LOG.info("Planning graph targets (names=%s)", names_set)

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    catalog = runtime_bundle.catalog
    targets = [t for t in catalog.all_targets if t.module == "graphs"]

    if names_set:
        targets = [t for t in targets if t.name in names_set]

    target_names = [t.name for t in targets]
    ordered = catalog.closure(tuple(target_names)) if target_names else ()

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


register_operation(
    OperationSpec(
        operation_id="graphs.targets.list",
        name="List Graph Targets",
        description="List graph build targets",
        handler=graph_targets_list_handler,
        group="graphs",
        require_runtime=True,
        require_gateway=True,
    )
)

register_operation(
    OperationSpec(
        operation_id="graphs.targets.plan",
        name="Graph Targets Plan",
        description="Display an execution plan for graph targets",
        handler=graph_targets_plan_handler,
        group="graphs",
        require_runtime=True,
        require_gateway=True,
    )
)

__all__ = [
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphTargetInfo",
    "GraphTargetsResult",
    "graph_targets_list_handler",
    "graph_targets_plan_handler",
]
