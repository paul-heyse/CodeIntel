"""Graph handlers.

Handlers for graph target listing and execution planning.
These handlers use the build registry instead of the legacy graph plugin registry.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    GraphPlanResult,
    GraphPlanStage,
    GraphTargetInfo,
    GraphTargetsResult,
)
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.dag_catalog import TargetDescriptor
    from codeintel.cli.context import CommandContext
    from codeintel.runtime.runtime_bundle import RuntimeBundle

LOG = logging.getLogger(__name__)


class SelectionPolicy(StrEnum):
    """Policy for handling unknown plugin names."""

    LENIENT = "lenient"
    STRICT = "strict"


class DependencyPolicy(StrEnum):
    """Policy for handling missing/implicit dependencies."""

    LENIENT = "lenient"
    STRICT = "strict"


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
    targets = _filter_graph_targets(runtime_bundle, names_set)

    target_infos = _build_target_infos(
        runtime_bundle,
        targets,
        description_prefix="Graph target",
    )

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
    targets = _filter_graph_targets(runtime_bundle, names_set)
    stages = _build_plan_stages(runtime_bundle, targets)

    return CliResult.ok(
        GraphPlanResult(
            stages=stages,
            total_targets=len(stages[0].targets) if stages else 0,
        )
    )


def graph_plugins_handler(
    ctx: CommandContext,
) -> CliResult[GraphPlanResult | GraphTargetsResult]:
    """List graph plugins or show an execution plan.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[GraphPlanResult | GraphTargetsResult]
        Planned execution order or plugin list.

    Raises
    ------
    ValueError
        If selection_policy is strict and unknown plugins are requested.
    """
    plan = ctx.params.get_bool("plan")
    names = ctx.params.get_list("names")
    selection_policy = ctx.params.get_enum(
        "selection_policy",
        SelectionPolicy,
        SelectionPolicy.LENIENT,
    )
    dependency_policy = ctx.params.get_enum(
        "dependency_policy",
        DependencyPolicy,
        DependencyPolicy.STRICT,
    )

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    names_set = set(names) if names else None

    catalog = runtime_bundle.catalog
    graph_targets = [t for t in catalog.all_targets if t.module == "graphs"]
    available_names = {t.name for t in graph_targets}

    if names_set:
        unknown = sorted(names_set - available_names)
        if unknown and selection_policy == SelectionPolicy.STRICT:
            msg = f"Unknown graph plugins: {unknown}"
            raise ValueError(msg)
        graph_targets = [t for t in graph_targets if t.name in names_set]

    if plan:
        target_names = [t.name for t in graph_targets]
        ordered = catalog.closure(tuple(target_names)) if target_names else ()
        if dependency_policy == DependencyPolicy.LENIENT:
            ordered = tuple(name for name in ordered if name in available_names)
        stages = [GraphPlanStage(stage=1, targets=list(ordered))]
        return CliResult.ok(GraphPlanResult(stages=stages, total_targets=len(ordered)))

    target_infos = _build_target_infos(
        runtime_bundle,
        graph_targets,
        description_prefix="Graph plugin",
    )
    return CliResult.ok(GraphTargetsResult(targets=target_infos, count=len(target_infos)))


def graph_targets_handler(
    ctx: CommandContext,
) -> CliResult[GraphPlanResult | GraphTargetsResult]:
    """List graph targets or show an execution plan.

    Returns
    -------
    CliResult[GraphPlanResult | GraphTargetsResult]
        Planned execution order or target list.
    """
    plan = ctx.params.get_bool("plan")
    names = ctx.params.get_list("names")
    names_set = set(names) if names else None

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    targets = _filter_graph_targets(runtime_bundle, names_set)

    if plan:
        stages = _build_plan_stages(runtime_bundle, targets)
        return CliResult.ok(
            GraphPlanResult(
                stages=stages,
                total_targets=len(stages[0].targets) if stages else 0,
            )
        )

    target_infos = _build_target_infos(
        runtime_bundle,
        targets,
        description_prefix="Graph target",
    )
    return CliResult.ok(GraphTargetsResult(targets=target_infos, count=len(target_infos)))


def _filter_graph_targets(
    runtime_bundle: RuntimeBundle,
    names_set: set[str] | None,
) -> list[TargetDescriptor]:
    catalog = runtime_bundle.catalog
    targets = [t for t in catalog.all_targets if t.module == "graphs"]
    if names_set:
        targets = [t for t in targets if t.name in names_set]
    return targets


def _build_target_infos(
    runtime_bundle: RuntimeBundle,
    targets: Iterable[TargetDescriptor],
    *,
    description_prefix: str,
) -> list[GraphTargetInfo]:
    catalog = runtime_bundle.catalog
    target_infos: list[GraphTargetInfo] = []
    for target in targets:
        name = target.name
        description = target.description
        dependencies = list(target.dependencies)
        tables = [output.key for output in catalog.table_outputs_by_target.get(name, ())]
        target_infos.append(
            GraphTargetInfo(
                name=name,
                description=description or f"{description_prefix}: {name}",
                dependencies=dependencies,
                tables=tables,
            )
        )
    return target_infos


def _build_plan_stages(
    runtime_bundle: RuntimeBundle,
    targets: Iterable[TargetDescriptor],
) -> list[GraphPlanStage]:
    catalog = runtime_bundle.catalog
    target_names = [
        getattr(target, "name", "") for target in targets if getattr(target, "name", "")
    ]
    ordered = catalog.closure(tuple(target_names)) if target_names else ()
    return [GraphPlanStage(stage=1, targets=list(ordered))]


__all__ = [
    "DependencyPolicy",
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphTargetInfo",
    "GraphTargetsResult",
    "SelectionPolicy",
    "graph_plugins_handler",
    "graph_targets_handler",
    "graph_targets_list_handler",
    "graph_targets_plan_handler",
]
