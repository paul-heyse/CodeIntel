"""Graph handlers.

Handlers for graph target listing and execution planning.
These handlers use the build registry instead of the legacy graph plugin registry.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.columnar import stream_from_items
from codeintel.cli.core.result_types import (
    GraphPlanResult,
    GraphPlanStage,
    GraphTargetInfo,
    TabularResult,
)
from codeintel.cli.handlers.runtime_helpers import (
    CliRuntimeComposeOptions,
    compose_cli_runtime_bundle,
)

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
) -> CliResult[TabularResult]:
    """List graph build targets.

    Parameters
    ----------
    ctx
        Command context with params:
        - names: Optional target names to filter.

    Returns
    -------
    CliResult[TabularResult]
        Stream of target records.
    """
    names_tuple = ctx.params.get_tuple("names")
    names_set: set[str] | None = set(names_tuple) if names_tuple else None

    LOG.info("Listing graph targets (names=%s)", names_set)

    runtime_bundle = compose_cli_runtime_bundle(
        runtime=ctx.runtime,
        gateway=ctx.gateway,
        options=CliRuntimeComposeOptions(verbosity=ctx.verbosity),
    )
    targets = _filter_graph_targets(runtime_bundle, names_set)

    target_infos = _build_target_infos(
        runtime_bundle,
        targets,
        description_prefix="Graph target",
    )

    stream = stream_from_items(target_infos)
    return CliResult.ok(
        TabularResult(
            stream=stream,
            metadata={"count": len(target_infos)},
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

    runtime_bundle = compose_cli_runtime_bundle(
        runtime=ctx.runtime,
        gateway=ctx.gateway,
        options=CliRuntimeComposeOptions(verbosity=ctx.verbosity),
    )
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
) -> CliResult[GraphPlanResult | TabularResult]:
    """List graph plugins or show an execution plan.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[GraphPlanResult | TabularResult]
        Planned execution order or plugin list.
    """
    plan = ctx.params.get_bool("plan")
    names = ctx.params.get_list("names")
    selection_policy = ctx.params.get_enum(
        "selection_policy",
        SelectionPolicy,
        SelectionPolicy.LENIENT,
    )
    if selection_policy is None:
        selection_policy = SelectionPolicy.LENIENT
    dependency_policy = ctx.params.get_enum(
        "dependency_policy",
        DependencyPolicy,
        DependencyPolicy.STRICT,
    )
    if dependency_policy is None:
        dependency_policy = DependencyPolicy.STRICT

    runtime_bundle = compose_cli_runtime_bundle(
        runtime=ctx.runtime,
        gateway=ctx.gateway,
        options=CliRuntimeComposeOptions(verbosity=ctx.verbosity),
    )
    graph_targets, available_names = _select_graph_plugin_targets(
        runtime_bundle,
        names,
        selection_policy,
    )
    if plan:
        return _plan_graph_plugins(
            runtime_bundle,
            graph_targets,
            available_names,
            dependency_policy,
        )

    return _list_graph_plugins(runtime_bundle, graph_targets)


def graph_targets_handler(
    ctx: CommandContext,
) -> CliResult[GraphPlanResult | TabularResult]:
    """List graph targets or show an execution plan.

    Returns
    -------
    CliResult[GraphPlanResult | TabularResult]
        Planned execution order or target list.
    """
    plan = ctx.params.get_bool("plan")
    names = ctx.params.get_list("names")
    names_set = set(names) if names else None

    runtime_bundle = compose_cli_runtime_bundle(
        runtime=ctx.runtime,
        gateway=ctx.gateway,
        options=CliRuntimeComposeOptions(verbosity=ctx.verbosity),
    )
    targets = _filter_graph_targets(runtime_bundle, names_set)

    if plan:
        stages = _build_plan_stages(runtime_bundle, targets)
        return CliResult[GraphPlanResult | TabularResult].ok(
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
    stream = stream_from_items(target_infos)
    return CliResult[GraphPlanResult | TabularResult].ok(
        TabularResult(
            stream=stream,
            metadata={"count": len(target_infos)},
        )
    )


def _filter_graph_targets(
    runtime_bundle: RuntimeBundle,
    names_set: set[str] | None,
) -> list[TargetDescriptor]:
    catalog = runtime_bundle.catalog
    targets = [t for t in catalog.all_targets if t.module == "graphs"]
    if names_set:
        targets = [t for t in targets if t.name in names_set]
    return targets


def _select_graph_plugin_targets(
    runtime_bundle: RuntimeBundle,
    names: list[str],
    selection_policy: SelectionPolicy,
) -> tuple[list[TargetDescriptor], set[str]]:
    graph_targets = [t for t in runtime_bundle.catalog.all_targets if t.module == "graphs"]
    available_names = {t.name for t in graph_targets}
    if names:
        names_set = set(names)
        unknown = sorted(names_set - available_names)
        if unknown and selection_policy == SelectionPolicy.STRICT:
            msg = f"Unknown graph plugins: {unknown}"
            raise ValueError(msg)
        graph_targets = [t for t in graph_targets if t.name in names_set]
    return graph_targets, available_names


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


def _plan_graph_plugins(
    runtime_bundle: RuntimeBundle,
    graph_targets: Iterable[TargetDescriptor],
    available_names: set[str],
    dependency_policy: DependencyPolicy,
) -> CliResult[GraphPlanResult | TabularResult]:
    target_names = [target.name for target in graph_targets]
    ordered = runtime_bundle.catalog.closure(tuple(target_names)) if target_names else ()
    if dependency_policy == DependencyPolicy.LENIENT:
        ordered = tuple(name for name in ordered if name in available_names)
    stages = [GraphPlanStage(stage=1, targets=list(ordered))]
    return CliResult[GraphPlanResult | TabularResult].ok(
        GraphPlanResult(stages=stages, total_targets=len(ordered))
    )


def _list_graph_plugins(
    runtime_bundle: RuntimeBundle,
    graph_targets: Iterable[TargetDescriptor],
) -> CliResult[GraphPlanResult | TabularResult]:
    target_infos = _build_target_infos(
        runtime_bundle,
        graph_targets,
        description_prefix="Graph plugin",
    )
    stream = stream_from_items(target_infos)
    return CliResult[GraphPlanResult | TabularResult].ok(
        TabularResult(
            stream=stream,
            metadata={"count": len(target_infos)},
        )
    )


__all__ = [
    "DependencyPolicy",
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphTargetInfo",
    "SelectionPolicy",
    "graph_plugins_handler",
    "graph_targets_handler",
    "graph_targets_list_handler",
    "graph_targets_plan_handler",
]
