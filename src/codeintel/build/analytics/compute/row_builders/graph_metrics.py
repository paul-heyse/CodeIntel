"""Row builders for graph metrics tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.query_results import coerce_optional_int

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

    from codeintel.build.analytics.compute.graphs import ComponentBundle, NeighborStats


@dataclass(frozen=True)
class FunctionGraphMetricInputs:
    """Inputs required to build graph_metrics_functions rows."""

    repo: str
    commit: str
    stats: NeighborStats
    centrality: Mapping[str, Mapping[int, float]]
    components: ComponentBundle
    graph_nodes: list[int]
    created_at: datetime


@dataclass(frozen=True)
class ModuleGraphMetricInputs:
    """Inputs required to build graph_metrics_modules rows."""

    repo: str
    commit: str
    modules: set[str]
    import_stats: NeighborStats
    centrality: Mapping[str, Mapping[str, float]]
    component_meta: Mapping[str, Mapping[str, int | bool]]
    symbol_inbound: Mapping[str, set[str]]
    symbol_outbound: Mapping[str, set[str]]
    created_at: datetime


def build_function_graph_metric_rows(
    inputs: FunctionGraphMetricInputs,
) -> list[dict[str, object]]:
    """Construct rows for analytics.graph_metrics_functions.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and ordering.

    Returns
    -------
    list[dict[str, object]]
        Row dicts ready for graph_metrics_functions insertion.
    """
    return [
        {
            "repo": inputs.repo,
            "commit": inputs.commit,
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
            "created_at": inputs.created_at,
        }
        for node in inputs.graph_nodes
    ]


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


def build_symbol_module_edges(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate symbol use edges to module-level adjacency.

    Parameters
    ----------
    symbol_use_edges
        Symbol use edges containing def_path/use_path values.
    module_by_path
        Mapping of file path to module name.

    Returns
    -------
    tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
        Modules involved plus inbound/outbound adjacency keyed by module.
    """
    modules: set[str] = set()
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    for record in symbol_use_edges:
        def_path = record.get("def_path")
        use_path = record.get("use_path")
        if def_path is None or use_path is None:
            continue
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None:
            continue
        modules.update((use_module, def_module))
        outbound[use_module].add(def_module)
        inbound[def_module].add(use_module)

    return modules, inbound, outbound


def build_module_graph_metric_rows(
    inputs: ModuleGraphMetricInputs,
) -> list[dict[str, object]]:
    """Construct rows for analytics.graph_metrics_modules.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and derived mappings.

    Returns
    -------
    list[dict[str, object]]
        Row dicts ready for graph_metrics_modules insertion.
    """
    return [
        {
            "repo": inputs.repo,
            "commit": inputs.commit,
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
                if (component_id := inputs.component_meta["component_id"].get(module)) is not None
                else None
            ),
            "import_layer": (
                int(layer_val)
                if (layer_val := inputs.component_meta["layer"].get(module)) is not None
                else None
            ),
            "symbol_fan_in": len(inputs.symbol_inbound.get(module, ())),
            "symbol_fan_out": len(inputs.symbol_outbound.get(module, ())),
            "created_at": inputs.created_at,
        }
        for module in sorted(inputs.modules)
    ]


__all__ = [
    "FunctionGraphMetricInputs",
    "ModuleGraphMetricInputs",
    "build_function_graph_metric_rows",
    "build_module_graph_metric_rows",
    "build_symbol_module_edges",
    "component_metadata_from_import_rows",
    "merge_component_metadata",
]
