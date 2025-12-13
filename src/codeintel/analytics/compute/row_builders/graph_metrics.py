"""Row builders for graph metrics tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import ibis
from ibis.common.exceptions import IbisError

from codeintel.config.datasets import (
    GraphMetricsFunctionsRow,
    GraphMetricsModulesRow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    import pandas as pd

    from codeintel.analytics.compute.graphs import ComponentBundle, NeighborStats
    from codeintel.config import GraphMetricsStepConfig
    from codeintel.storage.gateway import StorageGateway


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows into a list of dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Records returned by ``DataFrame.to_dict(orient="records")``.
    """
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))


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
) -> list[GraphMetricsFunctionsRow]:
    """Construct rows for analytics.graph_metrics_functions.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and ordering.

    Returns
    -------
    list[GraphMetricsFunctionsRow]
        Row dicts ready for graph_metrics_functions insertion.
    """
    return [
        GraphMetricsFunctionsRow(
            repo=inputs.cfg.repo,
            commit=inputs.cfg.commit,
            function_goid_h128=int(node),
            call_fan_in=len(inputs.stats.in_neighbors.get(node, ())),
            call_fan_out=len(inputs.stats.out_neighbors.get(node, ())),
            call_in_degree=inputs.stats.in_counts.get(node, 0),
            call_out_degree=inputs.stats.out_counts.get(node, 0),
            call_pagerank=inputs.centrality["pagerank"].get(node),
            call_betweenness=inputs.centrality["betweenness"].get(node),
            call_closeness=inputs.centrality["closeness"].get(node),
            call_cycle_member=inputs.components.in_cycle.get(node, False),
            call_cycle_id=inputs.components.scc_id.get(node),
            call_layer=inputs.components.layer.get(node),
            created_at=inputs.created_at,
        )
        for node in inputs.graph_nodes
    ]


def component_metadata_from_import_table(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, dict[str, int | bool]] | None:
    """Load cached import graph component metadata using Ibis.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    repo
        Repository slug anchoring the lookup.
    commit
        Commit hash anchoring the lookup.

    Returns
    -------
    dict[str, dict[str, int | bool]] | None
        Cached component metadata when present; otherwise ``None``.
    """
    try:
        tbl = gateway.ibis.table("graph.import_modules")
        expr = tbl.filter(cast("Any", (tbl.repo == repo) & (tbl.commit == commit))).select(
            "module", "scc_id", "component_size", "layer"
        )
        df = cast("pd.DataFrame", expr.execute())
    except IbisError:
        return None
    if df.empty:
        return None

    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    for record in _to_records(df):
        name = str(record["module"])
        scc_id = record["scc_id"]
        component_size = record["component_size"]
        layer = record["layer"]
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
    """Overlay cached component metadata on computed values when available.

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
    """Load symbol use edges aggregated to modules using Ibis.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    module_by_path
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
        su = gateway.ibis.table("graph.symbol_use_edges")
        m_def = gateway.ibis.table("core.modules").view()
        m_use = gateway.ibis.table("core.modules").view()

        def_module_present = cast("Any", ibis.coalesce(m_def.module, "")).length() > 0
        use_module_present = cast("Any", ibis.coalesce(m_use.module, "")).length() > 0
        joined = (
            su.left_join(m_def, cast("Any", su.def_path == m_def.path))
            .left_join(m_use, cast("Any", su.use_path == m_use.path))
            .filter(
                cast(
                    "Any",
                    def_module_present & use_module_present,
                )
            )
            .select(
                use_module=m_use.module,
                def_module=m_def.module,
            )
        )
        df = cast("pd.DataFrame", joined.execute())

        for record in _to_records(df):
            src = str(record["use_module"])
            dst = str(record["def_module"])
            modules.update((src, dst))
            outbound[src].add(dst)
            inbound[dst].add(src)
        return modules, inbound, outbound

    su = gateway.ibis.table("graph.symbol_use_edges")
    expr = su.select("def_path", "use_path")
    df = cast("pd.DataFrame", expr.execute())

    for record in _to_records(df):
        def_module = module_by_path.get(str(record["def_path"]))
        use_module = module_by_path.get(str(record["use_path"]))
        if def_module is None or use_module is None:
            continue
        modules.update((use_module, def_module))
        outbound[use_module].add(def_module)
        inbound[def_module].add(use_module)

    return modules, inbound, outbound


def build_module_graph_metric_rows(
    inputs: ModuleGraphMetricInputs,
) -> list[GraphMetricsModulesRow]:
    """Construct rows for analytics.graph_metrics_modules.

    Parameters
    ----------
    inputs
        Aggregated inputs capturing configuration, metrics, and derived mappings.

    Returns
    -------
    list[GraphMetricsModulesRow]
        Row dicts ready for graph_metrics_modules insertion.
    """
    return [
        GraphMetricsModulesRow(
            repo=inputs.cfg.repo,
            commit=inputs.cfg.commit,
            module=module,
            import_fan_in=len(inputs.import_stats.in_neighbors.get(module, ())),
            import_fan_out=len(inputs.import_stats.out_neighbors.get(module, ())),
            import_in_degree=inputs.import_stats.in_counts.get(module, 0),
            import_out_degree=inputs.import_stats.out_counts.get(module, 0),
            import_pagerank=inputs.centrality["pagerank"].get(module),
            import_betweenness=inputs.centrality["betweenness"].get(module),
            import_closeness=inputs.centrality["closeness"].get(module),
            import_cycle_member=bool(inputs.component_meta["in_cycle"].get(module, False)),
            import_cycle_id=(
                int(component_id)
                if (component_id := inputs.component_meta["component_id"].get(module)) is not None
                else None
            ),
            import_layer=(
                int(layer_val)
                if (layer_val := inputs.component_meta["layer"].get(module)) is not None
                else None
            ),
            symbol_fan_in=len(inputs.symbol_inbound.get(module, ())),
            symbol_fan_out=len(inputs.symbol_outbound.get(module, ())),
            created_at=inputs.created_at,
        )
        for module in sorted(inputs.modules)
    ]


__all__ = [
    "FunctionGraphMetricInputs",
    "ModuleGraphMetricInputs",
    "build_function_graph_metric_rows",
    "build_module_graph_metric_rows",
    "component_metadata_from_import_table",
    "load_symbol_module_edges",
    "merge_component_metadata",
]
