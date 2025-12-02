"""Row builders for extended graph metrics tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from codeintel.analytics.graph_service import GraphContext, to_decimal_id
from codeintel.config.datasets import (
    GraphMetricsFunctionsExtRow,
    GraphMetricsModulesExtRow,
)

# Aliases for backward compatibility
FunctionGraphMetricsExtRow = GraphMetricsFunctionsExtRow
ModuleGraphMetricsExtRow = GraphMetricsModulesExtRow


def _int_or_none(value: float | str | Decimal | None) -> int | None:
    """
    Return an integer when convertible, otherwise None.

    Returns
    -------
    int | None
        Converted integer or None when conversion fails.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def build_function_metric_ext_rows(
    inputs: FunctionMetricExtInputs,
) -> list[FunctionGraphMetricsExtRow]:
    """
    Construct rows for analytics.graph_metrics_functions_ext.

    Returns
    -------
    list[FunctionGraphMetricsExtRow]
        Rows ready for insertion into analytics.graph_metrics_functions_ext.
    """
    created_at = inputs.ctx.resolved_now()
    rows: list[FunctionGraphMetricsExtRow] = []
    for node in inputs.centralities["betweenness"]:
        goid_decimal = to_decimal_id(node)
        if goid_decimal is None:
            continue
        rows.append(
            FunctionGraphMetricsExtRow(
                repo=inputs.repo,
                commit=inputs.commit,
                function_goid_h128=int(goid_decimal),
                call_betweenness=float(inputs.centralities["betweenness"].get(node, 0.0)),
                call_closeness=float(inputs.centralities["closeness"].get(node, 0.0)),
                call_eigenvector=float(inputs.centralities["eigenvector"].get(node, 0.0)),
                call_harmonic=float(inputs.centralities["harmonic"].get(node, 0.0)),
                call_core_number=_int_or_none(inputs.structure["core_number"].get(node)),
                call_clustering_coeff=float(inputs.structure["clustering"].get(node, 0.0)),
                call_triangle_count=int(inputs.structure["triangles"].get(node, 0)),
                call_is_articulation=node in inputs.articulations,
                call_articulation_impact=None,
                call_is_bridge_endpoint=inputs.bridge_incident.get(node, 0) > 0,
                call_component_id=_int_or_none(inputs.components["component_id"].get(node)),
                call_component_size=_int_or_none(inputs.components["component_size"].get(node)),
                call_scc_id=_int_or_none(inputs.components["scc_id"].get(node)),
                call_scc_size=_int_or_none(inputs.components["scc_size"].get(node)),
                call_ancestor_count=_int_or_none(inputs.ancestor_count.get(node, 0)),
                call_descendant_count=_int_or_none(inputs.descendant_count.get(node, 0)),
                call_community_id=_int_or_none(inputs.structure["community_id"].get(node)),
                created_at=created_at,
            )
        )
    return rows


def build_module_metric_ext_rows(
    inputs: ModuleMetricExtInputs,
) -> list[ModuleGraphMetricsExtRow]:
    """
    Construct rows for analytics.graph_metrics_modules_ext.

    Returns
    -------
    list[ModuleGraphMetricsExtRow]
        Rows ready for insertion into analytics.graph_metrics_modules_ext.
    """
    created_at = inputs.ctx.resolved_now()
    return [
        ModuleGraphMetricsExtRow(
            repo=inputs.repo,
            commit=inputs.commit,
            module=module,
            import_betweenness=float(inputs.centralities["betweenness"].get(module, 0.0)),
            import_closeness=float(inputs.centralities["closeness"].get(module, 0.0)),
            import_eigenvector=float(inputs.centralities["eigenvector"].get(module, 0.0)),
            import_harmonic=float(inputs.centralities["harmonic"].get(module, 0.0)),
            import_k_core=_int_or_none(inputs.structure["core_number"].get(module)),
            import_constraint=float(inputs.structure["constraint"].get(module, 0.0)),
            import_effective_size=float(inputs.structure["effective_size"].get(module, 0.0)),
            import_rich_club=bool(inputs.rich_club.get(module, False)),
            import_shell_index=_int_or_none(inputs.structure["core_number"].get(module)),
            import_community_id=_int_or_none(inputs.structure["community_id"].get(module)),
            import_component_id=_int_or_none(inputs.components["component_id"].get(module)),
            import_component_size=_int_or_none(inputs.components["component_size"].get(module)),
            import_scc_id=_int_or_none(inputs.components["scc_id"].get(module)),
            import_scc_size=_int_or_none(inputs.components["scc_size"].get(module)),
            created_at=created_at,
        )
        for module in inputs.nodes
    ]


__all__ = [
    "FunctionMetricExtInputs",
    "ModuleMetricExtInputs",
    "build_function_metric_ext_rows",
    "build_module_metric_ext_rows",
]
