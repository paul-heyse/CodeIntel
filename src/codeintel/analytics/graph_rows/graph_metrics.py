"""Row builders for graph metrics tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from codeintel.analytics.graph_service import ComponentBundle, NeighborStats
from codeintel.config import GraphMetricsStepConfig
from codeintel.storage.gateway import DuckDBError, StorageGateway

FunctionMetricRow = tuple[
    str,
    str,
    int,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    float | None,
    bool,
    int | None,
    int | None,
    str,
]
ModuleMetricRow = tuple[
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    float | None,
    bool,
    int | None,
    int | None,
    int,
    int,
    str,
]


@dataclass(frozen=True)
class FunctionGraphMetricInputs:
    """Inputs required to build graph_metrics_functions rows."""

    cfg: GraphMetricsStepConfig
    stats: NeighborStats
    centrality: Mapping[str, Mapping[Any, float]]
    components: ComponentBundle
    graph_nodes: list[Any]
    created_at: datetime


@dataclass(frozen=True)
class ModuleGraphMetricInputs:
    """Inputs required to build graph_metrics_modules rows."""

    cfg: GraphMetricsStepConfig
    modules: set[str]
    import_stats: NeighborStats
    centrality: Mapping[str, Mapping[Any, float]]
    component_meta: Mapping[str, Mapping[Any, int | bool]]
    symbol_inbound: Mapping[str, set[str]]
    symbol_outbound: Mapping[str, set[str]]
    created_at: datetime


def build_function_graph_metric_rows(
    inputs: FunctionGraphMetricInputs,
) -> list[FunctionMetricRow]:
    """
    Construct rows for analytics.graph_metrics_functions.

    Parameters
    ----------
    inputs :
        Aggregated inputs capturing configuration, metrics, and ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples ready for graph_metrics_functions insertion.
    """
    return [
        (
            inputs.cfg.repo,
            inputs.cfg.commit,
            int(node),
            len(inputs.stats.in_neighbors.get(node, ())),
            len(inputs.stats.out_neighbors.get(node, ())),
            inputs.stats.in_counts.get(node, 0),
            inputs.stats.out_counts.get(node, 0),
            inputs.centrality["pagerank"].get(node),
            inputs.centrality["betweenness"].get(node),
            inputs.centrality["closeness"].get(node),
            inputs.components.in_cycle.get(node, False),
            inputs.components.scc_id.get(node),
            inputs.components.layer.get(node),
            inputs.created_at.isoformat(),
        )
        for node in inputs.graph_nodes
    ]


def component_metadata_from_import_table(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, dict[str, int | bool]] | None:
    """
    Load cached import graph component metadata if present.

    Parameters
    ----------
    gateway :
        Storage gateway providing the DuckDB connection.
    repo :
        Repository slug anchoring the lookup.
    commit :
        Commit hash anchoring the lookup.

    Returns
    -------
    dict[str, dict[str, int | bool]] | None
        Cached component metadata when present; otherwise ``None``.
    """
    try:
        rows = gateway.con.execute(
            """
            SELECT module, scc_id, component_size, layer
            FROM graph.import_modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except DuckDBError:
        return None
    if not rows:
        return None

    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    for module, scc_id, component_size, layer in rows:
        name = str(module)
        comp_id[name] = int(scc_id) if scc_id is not None else -1
        size = int(component_size) if component_size is not None else 0
        in_cycle[name] = size > 1
        if layer is not None:
            layer_by_module[name] = int(layer)
    return {
        "component_id": {node: int(val) for node, val in comp_id.items()},
        "in_cycle": {node: bool(flag) for node, flag in in_cycle.items()},
        "layer": {node: int(val) for node, val in layer_by_module.items()},
    }


def merge_component_metadata(
    graph_nodes: set[Any],
    computed: Mapping[str, Mapping[Any, int | bool]],
    cached: Mapping[str, Mapping[Any, int | bool]] | None,
) -> dict[str, dict[Any, int | bool]]:
    """
    Overlay cached component metadata on computed values when available.

    Returns
    -------
    dict[str, dict[Any, int | bool]]
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


def load_symbol_module_edges(
    gateway: StorageGateway,
    module_by_path: dict[str, str] | None,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """
    Load symbol use edges aggregated to modules.

    Parameters
    ----------
    gateway :
        Storage gateway providing the DuckDB connection.
    module_by_path :
        Optional mapping from file path to module name; when omitted, modules are
        resolved directly from the database.

    Returns
    -------
    tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
        Modules involved plus inbound/outbound adjacency keyed by module.
    """
    modules: set[str] = set()
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    if module_by_path is None:
        rows = gateway.con.execute(
            """
            SELECT m_use.module, m_def.module
            FROM graph.symbol_use_edges su
            LEFT JOIN core.modules m_def ON m_def.path = su.def_path
            LEFT JOIN core.modules m_use ON m_use.path = su.use_path
            WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
            """
        ).fetchall()

        for use_module, def_module in rows:
            src = str(use_module)
            dst = str(def_module)
            modules.update((src, dst))
            outbound[src].add(dst)
            inbound[dst].add(src)
        return modules, inbound, outbound

    path_rows = gateway.con.execute(
        "SELECT def_path, use_path FROM graph.symbol_use_edges"
    ).fetchall()
    for def_path, use_path in path_rows:
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
) -> list[ModuleMetricRow]:
    """
    Construct rows for analytics.graph_metrics_modules.

    Parameters
    ----------
    inputs :
        Aggregated inputs capturing configuration, metrics, and derived mappings.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples ready for graph_metrics_modules insertion.
    """
    return [
        (
            inputs.cfg.repo,
            inputs.cfg.commit,
            module,
            len(inputs.import_stats.in_neighbors.get(module, ())),
            len(inputs.import_stats.out_neighbors.get(module, ())),
            inputs.import_stats.in_counts.get(module, 0),
            inputs.import_stats.out_counts.get(module, 0),
            inputs.centrality["pagerank"].get(module),
            inputs.centrality["betweenness"].get(module),
            inputs.centrality["closeness"].get(module),
            bool(inputs.component_meta["in_cycle"].get(module, False)),
            (
                int(component_id)
                if (component_id := inputs.component_meta["component_id"].get(module)) is not None
                else None
            ),
            (
                int(layer_val)
                if (layer_val := inputs.component_meta["layer"].get(module)) is not None
                else None
            ),
            len(inputs.symbol_inbound.get(module, ())),
            len(inputs.symbol_outbound.get(module, ())),
            inputs.created_at.isoformat(),
        )
        for module in sorted(inputs.modules)
    ]


__all__ = [
    "build_function_graph_metric_rows",
    "build_module_graph_metric_rows",
    "component_metadata_from_import_table",
    "load_symbol_module_edges",
    "merge_component_metadata",
]
