"""Row builders for extended graph metrics tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from codeintel.analytics.graph_service import GraphContext, to_decimal_id

FunctionMetricExtRow = tuple[Any, ...]

ModuleMetricExtRow = tuple[Any, ...]


@dataclass(frozen=True)
class FunctionMetricExtInputs:
    """Inputs required to build function-level extended metric rows."""

    repo: str
    commit: str
    ctx: GraphContext
    centralities: Mapping[str, Mapping[Any, float]]
    structure: Mapping[str, Mapping[Any, float | int | bool]]
    components: Mapping[str, Mapping[Any, int]]
    articulations: set[int]
    bridge_incident: Mapping[int, int]
    ancestor_count: Mapping[int, int]
    descendant_count: Mapping[int, int]


@dataclass(frozen=True)
class ModuleMetricExtInputs:
    """Inputs required to build module-level extended metric rows."""

    repo: str
    commit: str
    ctx: GraphContext
    centralities: Mapping[str, Mapping[Any, float]]
    structure: Mapping[str, Mapping[Any, float | int]]
    components: Mapping[str, Mapping[Any, int]]
    rich_club: Mapping[Any, bool]
    nodes: list[str]


def build_function_metric_ext_rows(inputs: FunctionMetricExtInputs) -> list[FunctionMetricExtRow]:
    """
    Construct rows for analytics.graph_metrics_functions_ext.

    Returns
    -------
    list[FunctionMetricExtRow]
        Rows ready for insertion into analytics.graph_metrics_functions_ext.
    """
    created_at = inputs.ctx.resolved_now()
    return [
        (
            inputs.repo,
            inputs.commit,
            to_decimal_id(node),
            inputs.centralities["betweenness"].get(node, 0.0),
            inputs.centralities["closeness"].get(node, 0.0),
            inputs.centralities["eigenvector"].get(node, 0.0),
            inputs.centralities["harmonic"].get(node, 0.0),
            inputs.structure["core_number"].get(node),
            inputs.structure["clustering"].get(node, 0.0),
            int(inputs.structure["triangles"].get(node, 0)),
            node in inputs.articulations,
            None,
            inputs.bridge_incident.get(node, 0) > 0,
            inputs.components["component_id"].get(node),
            inputs.components["component_size"].get(node),
            inputs.components["scc_id"].get(node),
            inputs.components["scc_size"].get(node),
            inputs.ancestor_count.get(node, 0),
            inputs.descendant_count.get(node, 0),
            inputs.structure["community_id"].get(node),
            created_at,
        )
        for node in inputs.centralities["betweenness"]
    ]


def build_module_metric_ext_rows(inputs: ModuleMetricExtInputs) -> list[ModuleMetricExtRow]:
    """
    Construct rows for analytics.graph_metrics_modules_ext.

    Returns
    -------
    list[ModuleMetricExtRow]
        Rows ready for insertion into analytics.graph_metrics_modules_ext.
    """
    created_at = inputs.ctx.resolved_now()
    return [
        (
            inputs.repo,
            inputs.commit,
            module,
            inputs.centralities["betweenness"].get(module, 0.0),
            inputs.centralities["closeness"].get(module, 0.0),
            inputs.centralities["eigenvector"].get(module, 0.0),
            inputs.centralities["harmonic"].get(module, 0.0),
            inputs.structure["core_number"].get(module),
            inputs.structure["constraint"].get(module, 0.0),
            inputs.structure["effective_size"].get(module, 0.0),
            inputs.rich_club.get(module, False),
            inputs.structure["core_number"].get(module),
            inputs.structure["community_id"].get(module),
            inputs.components["component_id"].get(module),
            inputs.components["component_size"].get(module),
            inputs.components["scc_id"].get(module),
            inputs.components["scc_size"].get(module),
            created_at,
        )
        for module in inputs.nodes
    ]


__all__ = [
    "FunctionMetricExtInputs",
    "ModuleMetricExtInputs",
    "build_function_metric_ext_rows",
    "build_module_metric_ext_rows",
]
