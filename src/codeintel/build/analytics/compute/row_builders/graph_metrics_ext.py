"""Row builders for extended graph metrics tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.utilities.type_coercion import optional_int
from codeintel.core.data_models.ids import as_int

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.graphs.runtime.context import GraphContext


@dataclass(frozen=True)
class FunctionMetricExtInputs:
    """Inputs required to build function-level extended metric rows."""

    repo: str
    commit: str
    ctx: GraphContext
    centralities: Mapping[str, Mapping[int, float]]
    structure: Mapping[str, Mapping[int, float | int | bool]]
    components: Mapping[str, Mapping[int, int]]
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
    centralities: Mapping[str, Mapping[str, float]]
    structure: Mapping[str, Mapping[str, float | int]]
    components: Mapping[str, Mapping[str, int]]
    rich_club: Mapping[str, bool]
    nodes: list[str]


def build_function_metric_ext_rows(
    inputs: FunctionMetricExtInputs,
) -> list[dict[str, object]]:
    """Construct rows for analytics.graph_metrics_functions_ext.

    Returns
    -------
    list[dict[str, object]]
        Rows ready for insertion into analytics.graph_metrics_functions_ext.
    """
    created_at = inputs.ctx.resolved_now()
    rows: list[dict[str, object]] = []
    for node in inputs.centralities["betweenness"]:
        goid_value = as_int(node)
        if goid_value is None:
            continue
        rows.append(
            {
                "repo": inputs.repo,
                "commit": inputs.commit,
                "function_goid_h128": goid_value,
                "call_betweenness": float(inputs.centralities["betweenness"].get(node, 0.0)),
                "call_closeness": float(inputs.centralities["closeness"].get(node, 0.0)),
                "call_eigenvector": float(inputs.centralities["eigenvector"].get(node, 0.0)),
                "call_harmonic": float(inputs.centralities["harmonic"].get(node, 0.0)),
                "call_core_number": optional_int(inputs.structure["core_number"].get(node)),
                "call_clustering_coeff": float(inputs.structure["clustering"].get(node, 0.0)),
                "call_triangle_count": int(inputs.structure["triangles"].get(node, 0)),
                "call_is_articulation": node in inputs.articulations,
                "call_articulation_impact": None,
                "call_is_bridge_endpoint": inputs.bridge_incident.get(node, 0) > 0,
                "call_component_id": optional_int(inputs.components["component_id"].get(node)),
                "call_component_size": optional_int(inputs.components["component_size"].get(node)),
                "call_scc_id": optional_int(inputs.components["scc_id"].get(node)),
                "call_scc_size": optional_int(inputs.components["scc_size"].get(node)),
                "call_ancestor_count": optional_int(inputs.ancestor_count.get(node, 0)),
                "call_descendant_count": optional_int(inputs.descendant_count.get(node, 0)),
                "call_community_id": optional_int(inputs.structure["community_id"].get(node)),
                "created_at": created_at,
            }
        )
    return rows


def build_module_metric_ext_rows(
    inputs: ModuleMetricExtInputs,
) -> list[dict[str, object]]:
    """Construct rows for analytics.graph_metrics_modules_ext.

    Returns
    -------
    list[dict[str, object]]
        Rows ready for insertion into analytics.graph_metrics_modules_ext.
    """
    created_at = inputs.ctx.resolved_now()
    return [
        {
            "repo": inputs.repo,
            "commit": inputs.commit,
            "module": module,
            "import_betweenness": float(inputs.centralities["betweenness"].get(module, 0.0)),
            "import_closeness": float(inputs.centralities["closeness"].get(module, 0.0)),
            "import_eigenvector": float(inputs.centralities["eigenvector"].get(module, 0.0)),
            "import_harmonic": float(inputs.centralities["harmonic"].get(module, 0.0)),
            "import_k_core": optional_int(inputs.structure["core_number"].get(module)),
            "import_constraint": float(inputs.structure["constraint"].get(module, 0.0)),
            "import_effective_size": float(inputs.structure["effective_size"].get(module, 0.0)),
            "import_rich_club": bool(inputs.rich_club.get(module, False)),
            "import_shell_index": optional_int(inputs.structure["core_number"].get(module)),
            "import_community_id": optional_int(inputs.structure["community_id"].get(module)),
            "import_component_id": optional_int(inputs.components["component_id"].get(module)),
            "import_component_size": optional_int(inputs.components["component_size"].get(module)),
            "import_scc_id": optional_int(inputs.components["scc_id"].get(module)),
            "import_scc_size": optional_int(inputs.components["scc_size"].get(module)),
            "created_at": created_at,
        }
        for module in inputs.nodes
    ]


__all__ = [
    "FunctionMetricExtInputs",
    "ModuleMetricExtInputs",
    "build_function_metric_ext_rows",
    "build_module_metric_ext_rows",
]
