"""Row builders for graph metrics tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders.context import RowBuildContext
from codeintel.build.analytics.compute.row_builders.core import buffer_for_table
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.query_results import coerce_optional_int

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.analytics.compute.graphs import ComponentBundle, NeighborStats


@dataclass(frozen=True)
class FunctionGraphMetricInputs:
    """Inputs required to build graph_metrics_functions rows."""

    row_context: RowBuildContext
    stats: NeighborStats
    centrality: Mapping[str, Mapping[int, float]]
    components: ComponentBundle
    graph_nodes: list[int]


@dataclass(frozen=True)
class ModuleGraphMetricInputs:
    """Inputs required to build graph_metrics_modules rows."""

    row_context: RowBuildContext
    modules: set[str]
    import_stats: NeighborStats
    centrality: Mapping[str, Mapping[str, float]]
    component_meta: Mapping[str, Mapping[str, int | bool]]
    symbol_inbound: Mapping[str, set[str]]
    symbol_outbound: Mapping[str, set[str]]


def build_function_graph_metric_rows(
    inputs: FunctionGraphMetricInputs,
) -> ColumnarRowBuffer:
    """Construct rows for analytics.graph_metrics_functions.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and ordering.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for graph_metrics_functions insertion.
    """
    buffer = buffer_for_table("analytics.graph_metrics_functions")
    for node in inputs.graph_nodes:
        buffer.append(
            {
                "repo": inputs.row_context.repo,
                "commit": inputs.row_context.commit,
                "function_goid_h128": int(node),
                "call_fan_in": len(inputs.stats.in_neighbors.get(node, ())),
                "call_fan_out": len(inputs.stats.out_neighbors.get(node, ())),
                "call_in_degree": inputs.stats.in_counts.get(node, 0),
                "call_out_degree": inputs.stats.out_counts.get(node, 0),
                "call_pagerank": inputs.centrality["pagerank"].get(node),
                "call_betweenness": inputs.centrality["betweenness"].get(node),
                "call_closeness": inputs.centrality["closeness"].get(node),
                "call_cycle_member": inputs.components.in_cycle.get(node, False),
                "call_cycle_id": inputs.components.scc_id.get(node),
                "call_layer": inputs.components.layer.get(node),
                "created_at": inputs.row_context.created_at,
            }
        )
    return buffer


def component_metadata_from_import_rows(
    rows: Iterable[Mapping[str, object]],
) -> Mapping[str, Mapping[str, int | bool]] | None:
    """Build cached import graph metadata from pre-scoped rows.

    Parameters
    ----------
    rows
        Import module rows containing module, scc_id, component_size, and layer.

    Returns
    -------
    dict[str, dict[str, int | bool]] | None
        Cached component metadata when present; otherwise ``None``.
    """
    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    found = False
    for record in rows:
        name = record.get("module")
        if name is None:
            continue
        found = True
        module = str(name)
        scc_id = coerce_optional_int(record.get("scc_id"), ctx="scc_id")
        component_size = coerce_optional_int(record.get("component_size"), ctx="component_size")
        layer = coerce_optional_int(record.get("layer"), ctx="layer")
        comp_id[module] = scc_id if scc_id is not None else -1
        size = component_size or 0
        in_cycle[module] = size > 1
        if layer is not None:
            layer_by_module[module] = layer

    if not found:
        return None

    return {
        "component_id": {node: int(val) for node, val in comp_id.items()},
        "in_cycle": {node: bool(flag) for node, flag in in_cycle.items()},
        "layer": {node: int(val) for node, val in layer_by_module.items()},
    }


def merge_component_metadata(
    graph_nodes: set[str],
    computed: Mapping[str, Mapping[str, int | bool]],
    cached: Mapping[str, Mapping[str, int | bool]] | None,
) -> dict[str, dict[str, int | bool]]:
    """Overlay cached component metadata on computed values when available.

    Returns
    -------
    dict[str, dict[str, int | bool]]
        Component metadata combining computed and cached values.
    """
    if cached is None:
        return {
            "component_id": dict(computed["component_id"]),
            "in_cycle": dict(computed["in_cycle"]),
            "layer": dict(computed["layer"]),
        }
    ids = dict(computed["component_id"])
    in_cycle = dict(computed["in_cycle"])
    layer = dict(computed["layer"])
    for node in graph_nodes:
        if node in cached.get("component_id", {}):
            ids[node] = cached["component_id"][node]
            in_cycle[node] = bool(cached["in_cycle"].get(node, False))
            layer[node] = int(cached["layer"].get(node, layer.get(node, 0)))
    return {"component_id": ids, "in_cycle": in_cycle, "layer": layer}


def build_module_graph_metric_rows(
    inputs: ModuleGraphMetricInputs,
) -> ColumnarRowBuffer:
    """Construct rows for analytics.graph_metrics_modules.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and derived mappings.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for graph_metrics_modules insertion.
    """
    buffer = buffer_for_table("analytics.graph_metrics_modules")
    for module in inputs.modules:
        buffer.append(
            {
                "repo": inputs.row_context.repo,
                "commit": inputs.row_context.commit,
                "module": module,
                "import_fan_in": len(inputs.import_stats.in_neighbors.get(module, ())),
                "import_fan_out": len(inputs.import_stats.out_neighbors.get(module, ())),
                "import_in_degree": inputs.import_stats.in_counts.get(module, 0),
                "import_out_degree": inputs.import_stats.out_counts.get(module, 0),
                "import_pagerank": inputs.centrality["pagerank"].get(module),
                "import_betweenness": inputs.centrality["betweenness"].get(module),
                "import_closeness": inputs.centrality["closeness"].get(module),
                "import_cycle_member": bool(inputs.component_meta["in_cycle"].get(module, False)),
                "import_cycle_id": (
                    int(component_id)
                    if (component_id := inputs.component_meta["component_id"].get(module))
                    is not None
                    else None
                ),
                "import_layer": (
                    int(layer_val)
                    if (layer_val := inputs.component_meta["layer"].get(module)) is not None
                    else None
                ),
                "symbol_fan_in": len(inputs.symbol_inbound.get(module, ())),
                "symbol_fan_out": len(inputs.symbol_outbound.get(module, ())),
                "created_at": inputs.row_context.created_at,
            }
        )
    return buffer


__all__ = [
    "FunctionGraphMetricInputs",
    "ModuleGraphMetricInputs",
    "build_function_graph_metric_rows",
    "build_module_graph_metric_rows",
    "component_metadata_from_import_rows",
    "merge_component_metadata",
]
