"""Shared helpers to materialize Parquet-backed graphs as NetworkX views.

This module provides functions to load various graph types from
Parquet datasets into NetworkX graph structures. View-registry
fallthrough is intentionally disallowed in this layer.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import cast

import networkx as nx
import polars as pl

from codeintel.build.graphs.engine.datasets import (
    scan_snapshot_lazyframe,
    scan_snapshot_reader,
)
from codeintel.core.data_models.ids import as_int
from codeintel.core.data_models.ids import normalize_decimal_id as normalize_decimal
from codeintel.core.query_results import coerce_optional_float, iter_tuples_from_arrow_reader

log = logging.getLogger(__name__)


def _ensure_dataset_root(dataset_root: Path | None, table_key: str) -> Path | None:
    if dataset_root is None:
        log.warning("Dataset root not configured; cannot load %s", table_key)
        return None
    return dataset_root


def _filter_optional_scope(frame: pl.LazyFrame, *, repo: str, commit: str) -> pl.LazyFrame:
    available = set(frame.columns)
    if "repo" in available:
        frame = frame.filter(pl.col("repo").is_null() | (pl.col("repo") == repo))
    if "commit" in available:
        frame = frame.filter(pl.col("commit").is_null() | (pl.col("commit") == commit))
    return frame


def _maybe_to_gpu_graph(graph: nx.Graph, *, use_gpu: bool) -> nx.Graph:
    """
    Optionally shift a NetworkX graph toward a GPU backend.

    Parameters
    ----------
    graph : nx.Graph
        Graph instance to optionally prepare for GPU execution.
    use_gpu : bool
        Whether GPU execution was requested.

    Returns
    -------
    nx.Graph
        The original graph or a GPU-backed equivalent.
    """
    if not use_gpu:
        return graph

    try:
        importlib.import_module("nx_cugraph")
    except ImportError:
        log.debug("nx_cugraph not installed; leaving graph on CPU backend.")
        return graph

    log.debug("GPU backend requested; relying on nx_cugraph backend dispatch.")
    return graph


def _coerce_edge_weight_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _coerce_edge_weight_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def module_attrs_from_row(
    module: object,
    scc_id: object | None,
    component_size: object | None,
    layer: object | None,
) -> tuple[str, dict[str, int]]:
    """
    Build a normalized node attribute mapping for an import module row.

    Parameters
    ----------
    module :
        Module identifier from the import_modules table.
    scc_id :
        Strongly connected component identifier.
    component_size :
        Size of the SCC.
    layer :
        Condensation DAG layer.

    Returns
    -------
    tuple[str, dict[str, int]]
        Normalized module name and attribute dictionary.
    """
    module_name = str(module)
    attrs: dict[str, int] = {}
    scc_value = as_int(scc_id)
    if scc_value is not None:
        attrs["scc_id"] = scc_value
    comp_size_value = as_int(component_size)
    if comp_size_value is not None:
        attrs["component_size"] = comp_size_value
    layer_value = as_int(layer)
    if layer_value is not None:
        attrs["layer"] = layer_value
    return module_name, attrs


def load_call_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.DiGraph:
    """
    Build a call graph `DiGraph` of caller -> callee edges.

    Nodes are GOID integers; parallel edges are aggregated via `weight`.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.DiGraph
        Directed call graph with weighted edges and isolated nodes preserved.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.call_graph_edges")
    if dataset_root is None:
        return nx.DiGraph()
    reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="graph.call_graph_edges",
        snapshot_id=commit,
        columns=("caller_goid_h128", "callee_goid_h128"),
        repo=repo,
        commit=commit,
    )
    if reader is None:
        return nx.DiGraph()

    graph = nx.DiGraph()
    for caller_raw, callee_raw in iter_tuples_from_arrow_reader(reader):
        caller = normalize_decimal(caller_raw)
        callee = normalize_decimal(callee_raw)
        if caller is None or callee is None:
            continue
        if graph.has_edge(caller, callee):
            attrs = graph[caller][callee]
            attrs["weight"] = _coerce_edge_weight_int(attrs.get("weight"), default=0) + 1
        else:
            graph.add_edge(caller, callee, weight=1)

    node_reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="graph.call_graph_nodes",
        snapshot_id=commit,
        columns=("goid_h128", "kind"),
        repo=repo,
        commit=commit,
    )
    if node_reader is not None:
        for node_raw, kind in iter_tuples_from_arrow_reader(node_reader):
            node = normalize_decimal(node_raw)
            if node is None:
                continue
            if node in graph:
                continue
            attrs: dict[str, object] = {}
            if kind is not None:
                attrs["kind"] = str(kind)
            graph.add_node(node, **attrs)

    return cast("nx.DiGraph", _maybe_to_gpu_graph(graph, use_gpu=use_gpu))


def load_import_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.DiGraph:
    """
    Build a directed import graph `DiGraph` of module -> module edges.

    Edge weights represent aggregated edge counts when multiple edges exist.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.DiGraph
        Directed import graph with weights capturing edge multiplicity.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.import_graph_edges")
    if dataset_root is None:
        return nx.DiGraph()
    edge_reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="graph.import_graph_edges",
        snapshot_id=commit,
        columns=("src_module", "dst_module", "module_layer"),
        repo=repo,
        commit=commit,
    )
    if edge_reader is None:
        return nx.DiGraph()

    graph = nx.DiGraph()
    fallback_layer_by_module: dict[str, int] = {}
    for src, dst, layer in iter_tuples_from_arrow_reader(edge_reader):
        if src is None or dst is None:
            continue
        source = str(src)
        target = str(dst)
        layer_value = as_int(layer)
        if layer_value is not None:
            fallback_layer_by_module[source] = layer_value
        edge_data = graph.get_edge_data(source, target)
        weight = _coerce_edge_weight_int(edge_data.get("weight") if edge_data is not None else None)
        graph.add_edge(source, target, weight=weight + 1)

    module_reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="graph.import_modules",
        snapshot_id=commit,
        columns=("module", "scc_id", "component_size", "layer"),
        repo=repo,
        commit=commit,
    )
    if module_reader is not None:
        for module_row in iter_tuples_from_arrow_reader(module_reader):
            module_name, attrs = module_attrs_from_row(*module_row)
            graph.add_node(module_name, **attrs)
    elif fallback_layer_by_module:
        graph.add_nodes_from(
            [(module, {"layer": layer}) for module, layer in fallback_layer_by_module.items()]
        )
    return cast("nx.DiGraph", _maybe_to_gpu_graph(graph, use_gpu=use_gpu))


def load_test_function_bipartite(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.Graph:
    """
    Build a bipartite graph of tests <-> functions from coverage edges.

    Test nodes are keyed as ("t", test_id); function nodes as ("f", goid).
    Edge weight is derived from coverage_ratio when present.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.Graph
        Undirected bipartite graph with weighted coverage edges.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "analytics.test_coverage_edges")
    if dataset_root is None:
        return nx.Graph()
    reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="analytics.test_coverage_edges",
        snapshot_id=commit,
        columns=("test_id", "function_goid_h128", "coverage_ratio"),
        repo=repo,
        commit=commit,
    )
    if reader is None:
        return nx.Graph()

    graph = nx.Graph()
    for test_id, goid_raw, coverage_ratio in iter_tuples_from_arrow_reader(reader):
        goid = normalize_decimal(goid_raw)
        if test_id is None or goid is None:
            continue
        test_node = ("t", str(test_id))
        func_node = ("f", goid)
        if not graph.has_node(test_node):
            graph.add_node(test_node, bipartite=0)
        if not graph.has_node(func_node):
            graph.add_node(func_node, bipartite=1)
        weight = coerce_optional_float(coverage_ratio, ctx="coverage_ratio") or 0.0
        if graph.has_edge(test_node, func_node):
            attrs = graph[test_node][func_node]
            attrs["weight"] = _coerce_edge_weight_float(attrs.get("weight")) + weight
        else:
            graph.add_edge(test_node, func_node, weight=weight)
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)


def parse_reference_modules(ref_modules: object, allowed_modules: set[str]) -> list[str]:
    """Normalize reference modules input into a filtered list.

    Returns
    -------
    list[str]
        Allowed module names parsed from input.
    """
    modules: list[str] = []
    if isinstance(ref_modules, list):
        modules = [str(mod) for mod in ref_modules]
    elif isinstance(ref_modules, str):
        try:
            parsed = json.loads(ref_modules)
            if isinstance(parsed, list):
                modules = [str(mod) for mod in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            modules = []
    if allowed_modules:
        return [module for module in modules if module in allowed_modules]
    return modules


def load_config_module_bipartite(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.Graph:
    """
    Build a bipartite graph of config keys <-> modules.

    Keys are ("c", key); modules are ("m", module). Edge weight equals one per
    reference occurrence.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.Graph
        Undirected bipartite graph for configuration references.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "analytics.config_values")
    if dataset_root is None:
        return nx.Graph()
    modules_frame = scan_snapshot_lazyframe(
        dataset_root=dataset_root,
        table_key="core.modules",
        snapshot_id=commit,
        columns=("module", "repo", "commit"),
        repo=None,
        commit=None,
    )
    if modules_frame is None:
        return nx.Graph()
    modules_frame = _filter_optional_scope(modules_frame, repo=repo, commit=commit)
    module_rows = modules_frame.select("module").collect()
    allowed_modules = {str(mod) for mod in module_rows.get_column("module").to_list()}

    config_frame = scan_snapshot_lazyframe(
        dataset_root=dataset_root,
        table_key="analytics.config_values",
        snapshot_id=commit,
        columns=("key", "reference_modules", "repo", "commit"),
        repo=repo,
        commit=commit,
    )
    if config_frame is None:
        return nx.Graph()

    graph = nx.Graph()
    total_rows = 0
    empty_refs = 0
    parsed_modules = 0
    kept_modules = 0
    dropped_modules = 0
    for row in config_frame.collect().iter_rows(named=True):
        total_rows += 1
        key = row.get("key")
        ref_modules = row.get("reference_modules")
        if key is None or ref_modules is None:
            empty_refs += 1
            continue
        key_node = ("c", str(key))
        if not graph.has_node(key_node):
            graph.add_node(key_node, bipartite=0)

        raw_modules = parse_reference_modules(ref_modules, set())
        parsed_modules += len(raw_modules)
        filtered_modules = (
            [module for module in raw_modules if module in allowed_modules]
            if allowed_modules
            else raw_modules
        )
        if allowed_modules and raw_modules and not filtered_modules:
            filtered_modules = raw_modules
        kept_modules += len(filtered_modules)
        dropped_modules += len(raw_modules) - len(filtered_modules)

        for module_name in filtered_modules:
            module_node = ("m", module_name)
            if not graph.has_node(module_node):
                graph.add_node(module_node, bipartite=1)
            if graph.has_edge(key_node, module_node):
                attrs = graph[key_node][module_node]
                attrs["weight"] = _coerce_edge_weight_int(attrs.get("weight"), default=0) + 1
            else:
                graph.add_edge(key_node, module_node, weight=1)
    log.info(
        "Config bipartite built: rows=%d empty_refs=%d allowed_modules=%d "
        "parsed_modules=%d kept_modules=%d dropped_modules=%d graph_nodes=%d edges=%d",
        total_rows,
        empty_refs,
        len(allowed_modules),
        parsed_modules,
        kept_modules,
        dropped_modules,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)


def load_symbol_module_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.Graph:
    """
    Build an undirected weighted graph of module-level symbol coupling.

    Edge weights count shared symbol def/use pairs between modules.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.Graph
        Undirected graph where weights reflect shared symbol relations.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return nx.Graph()
    edges_frame = scan_snapshot_lazyframe(
        dataset_root=dataset_root,
        table_key="graph.symbol_use_edges",
        snapshot_id=commit,
        columns=("def_path", "use_path"),
        repo=None,
        commit=None,
    )
    if edges_frame is None:
        return nx.Graph()
    modules_frame = scan_snapshot_lazyframe(
        dataset_root=dataset_root,
        table_key="core.modules",
        snapshot_id=commit,
        columns=("path", "module", "repo", "commit"),
        repo=None,
        commit=None,
    )
    if modules_frame is None:
        return nx.Graph()
    modules_frame = _filter_optional_scope(modules_frame, repo=repo, commit=commit).select(
        ["path", "module"]
    )
    def_modules = modules_frame.rename({"path": "def_path", "module": "def_module"})
    use_modules = modules_frame.rename({"path": "use_path", "module": "use_module"})
    joined = edges_frame.join(def_modules, on="def_path", how="left").join(
        use_modules, on="use_path", how="left"
    )

    graph = nx.Graph()
    for row in joined.collect().iter_rows(named=True):
        use_module = row.get("use_module")
        def_module = row.get("def_module")
        if use_module is None or def_module is None:
            continue
        left = str(use_module)
        right = str(def_module)
        if left == right:
            continue
        if graph.has_edge(left, right):
            attrs = graph[left][right]
            attrs["weight"] = _coerce_edge_weight_int(attrs.get("weight"), default=0) + 1
        else:
            graph.add_edge(left, right, weight=1)
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)


def load_symbol_function_graph(
    dataset_root: Path | None,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.Graph:
    """
    Build an undirected weighted graph of function-level symbol coupling (GOIDs).

    Edge weights count shared symbol def/use pairs between functions when available.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    commit : str
        Commit hash anchoring the snapshot.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    nx.Graph
        Undirected graph linking functions by shared symbol usage.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return nx.Graph()
    reader = scan_snapshot_reader(
        dataset_root=dataset_root,
        table_key="graph.symbol_use_edges",
        snapshot_id=commit,
        columns=("def_goid_h128", "use_goid_h128"),
    )
    if reader is None:
        return nx.Graph()

    graph = nx.Graph()
    for def_goid, use_goid in iter_tuples_from_arrow_reader(reader):
        if def_goid is None or use_goid is None:
            continue
        left = normalize_decimal(def_goid)
        right = normalize_decimal(use_goid)
        if left is None or right is None or left == right:
            continue
        if graph.has_edge(left, right):
            attrs = graph[left][right]
            attrs["weight"] = _coerce_edge_weight_int(attrs.get("weight"), default=0) + 1
        else:
            graph.add_edge(left, right, weight=1)
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)


__all__ = [
    "as_int",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "load_test_function_bipartite",
    "module_attrs_from_row",
    "normalize_decimal",
    "parse_reference_modules",
]
