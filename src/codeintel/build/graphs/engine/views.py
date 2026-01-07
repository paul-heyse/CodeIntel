"""Shared helpers to materialize Parquet-backed graphs as NetworkX views.

This module provides functions to load various graph types from
Parquet datasets into NetworkX graph structures. View-registry
fallthrough is intentionally disallowed in this layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.graphs.builders import EdgeWeightPolicy, add_weighted_edge
from codeintel.build.graphs.engine.datasets import (
    SnapshotScanRequest,
    scan_snapshot_reader,
    scan_snapshot_table,
)
from codeintel.build.graphs.rx import RxGraphStore, rx_to_networkx
from codeintel.build.tabular.conversion import table_to_reader
from codeintel.core.columnar.type_normalization import (
    normalize_binary_view_table,
    normalize_string_view_table,
)
from codeintel.core.data_models.ids import as_int
from codeintel.core.data_models.ids import normalize_decimal_id as normalize_decimal
from codeintel.core.query_results import iter_tuples_from_arrow_reader

if TYPE_CHECKING:
    import networkx as nx
    import pyarrow as pa

log = logging.getLogger(__name__)
_EDGE_WEIGHT_POLICY = EdgeWeightPolicy()


def _ensure_dataset_root(dataset_root: Path | None, table_key: str) -> Path | None:
    if dataset_root is None:
        log.warning("Dataset root not configured; cannot load %s", table_key)
        return None
    return dataset_root


def _scan_snapshot_reader(request: SnapshotScanRequest) -> pa.RecordBatchReader | None:
    return scan_snapshot_reader(request)


def _column_index(names: list[str], column: str) -> int | None:
    try:
        return names.index(column)
    except ValueError:
        return None


def _iter_scoped_rows(
    reader: pa.RecordBatchReader,
    *,
    repo: str,
    commit: str,
) -> Iterable[tuple[object, ...]]:
    names = list(reader.schema.names)
    repo_idx = _column_index(names, "repo")
    commit_idx = _column_index(names, "commit")
    for row in iter_tuples_from_arrow_reader(reader):
        if repo_idx is not None:
            row_repo = row[repo_idx]
            if row_repo is not None and str(row_repo) != repo:
                continue
        if commit_idx is not None:
            row_commit = row[commit_idx]
            if row_commit is not None and str(row_commit) != commit:
                continue
        yield row


def _normalize_view_table(table: pa.Table) -> pa.Table:
    return normalize_binary_view_table(normalize_string_view_table(table))


def _empty_graph(*, directed: bool) -> nx.Graph:
    store = RxGraphStore.directed() if directed else RxGraphStore.undirected()
    return rx_to_networkx(store.graph)


def _module_name_map(
    reader: object,
    *,
    repo: str,
    commit: str,
) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    specificity_by_path: dict[str, int] = {}
    for path, module, row_repo, row_commit in iter_tuples_from_arrow_reader(reader):
        if path is None or module is None:
            continue
        if row_repo is not None and str(row_repo) != repo:
            continue
        if row_commit is not None and str(row_commit) != commit:
            continue
        specificity = int(row_repo is not None) + int(row_commit is not None)
        key = str(path)
        if specificity < specificity_by_path.get(key, -1):
            continue
        module_by_path[key] = str(module)
        specificity_by_path[key] = specificity
    return module_by_path


def _add_call_edges(store: RxGraphStore, reader: object) -> None:
    for caller_raw, callee_raw in iter_tuples_from_arrow_reader(reader):
        caller = normalize_decimal(caller_raw)
        callee = normalize_decimal(callee_raw)
        if caller is None or callee is None:
            continue
        add_weighted_edge(store, caller, callee, policy=_EDGE_WEIGHT_POLICY)


def _add_call_nodes(store: RxGraphStore, reader: object) -> None:
    for node_raw, kind in iter_tuples_from_arrow_reader(reader):
        node = normalize_decimal(node_raw)
        if node is None:
            continue
        attrs: dict[str, object] = {}
        if kind is not None:
            attrs["kind"] = str(kind)
        store.set_node_attrs(node, attrs)


def _maybe_to_gpu_graph(graph: nx.Graph, *, use_gpu: bool) -> nx.Graph:
    """
    No-op for rustworkx-backed execution (CPU-only).

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
    if use_gpu:
        log.debug("GPU backend requested; rustworkx execution is CPU-only.")
    return graph


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
        return cast("nx.DiGraph", _empty_graph(directed=True))
    edge_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.call_graph_edges",
            snapshot_id=commit,
            columns=("caller_goid_h128", "callee_goid_h128"),
            repo=repo,
            commit=commit,
        )
    )
    if edge_table is None:
        return cast("nx.DiGraph", _empty_graph(directed=True))

    store = RxGraphStore.directed()
    _add_call_edges(store, table_to_reader(_normalize_view_table(edge_table)))

    node_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.call_graph_nodes",
            snapshot_id=commit,
            columns=("goid_h128", "kind"),
            repo=repo,
            commit=commit,
        )
    )
    if node_table is not None:
        _add_call_nodes(store, table_to_reader(_normalize_view_table(node_table)))

    graph = rx_to_networkx(store.graph)
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
        return cast("nx.DiGraph", _empty_graph(directed=True))
    edge_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.import_graph_edges",
            snapshot_id=commit,
            columns=("src_module", "dst_module", "module_layer"),
            repo=repo,
            commit=commit,
        )
    )
    if edge_table is None:
        return cast("nx.DiGraph", _empty_graph(directed=True))

    store = RxGraphStore.directed()
    fallback_layer_by_module: dict[str, int] = {}
    for src, dst, layer in iter_tuples_from_arrow_reader(
        table_to_reader(_normalize_view_table(edge_table))
    ):
        if src is None or dst is None:
            continue
        source = str(src)
        target = str(dst)
        layer_value = as_int(layer)
        if layer_value is not None:
            fallback_layer_by_module[source] = layer_value
        add_weighted_edge(store, source, target, policy=_EDGE_WEIGHT_POLICY)

    module_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.import_modules",
            snapshot_id=commit,
            columns=("module", "scc_id", "component_size", "layer"),
            repo=repo,
            commit=commit,
        )
    )
    if module_table is not None:
        for module_row in iter_tuples_from_arrow_reader(
            table_to_reader(_normalize_view_table(module_table))
        ):
            module_name, attrs = module_attrs_from_row(*module_row)
            store.set_node_attrs(module_name, attrs)
    elif fallback_layer_by_module:
        for module, layer in fallback_layer_by_module.items():
            store.set_node_attrs(module, {"layer": layer})
    graph = rx_to_networkx(store.graph)
    return cast("nx.DiGraph", _maybe_to_gpu_graph(graph, use_gpu=use_gpu))


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


@dataclass
class ConfigGraphStats:
    total_rows: int = 0
    empty_refs: int = 0
    parsed_modules: int = 0
    kept_modules: int = 0
    dropped_modules: int = 0


def _allowed_modules_from_reader(
    modules_reader: pa.RecordBatchReader,
    *,
    repo: str,
    commit: str,
) -> set[str]:
    names = list(modules_reader.schema.names)
    module_idx = _column_index(names, "module")
    if module_idx is None:
        return set()
    allowed: set[str] = set()
    for row in _iter_scoped_rows(modules_reader, repo=repo, commit=commit):
        value = row[module_idx]
        if value is None:
            continue
        allowed.add(str(value))
    return allowed


def _populate_config_graph(
    store: RxGraphStore,
    config_reader: pa.RecordBatchReader,
    *,
    repo: str,
    commit: str,
    allowed_modules: set[str],
) -> ConfigGraphStats:
    stats = ConfigGraphStats()
    names = list(config_reader.schema.names)
    key_idx = _column_index(names, "key")
    ref_idx = _column_index(names, "reference_modules")
    if key_idx is None or ref_idx is None:
        return stats
    for row in _iter_scoped_rows(config_reader, repo=repo, commit=commit):
        stats.total_rows += 1
        key = row[key_idx]
        ref_modules = row[ref_idx]
        if key is None or ref_modules is None:
            stats.empty_refs += 1
            continue
        key_node = ("c", str(key))
        store.set_node_attrs(key_node, {"bipartite": 0})

        raw_modules = parse_reference_modules(ref_modules, set())
        stats.parsed_modules += len(raw_modules)
        filtered_modules = (
            [module for module in raw_modules if module in allowed_modules]
            if allowed_modules
            else raw_modules
        )
        if allowed_modules and raw_modules and not filtered_modules:
            filtered_modules = raw_modules
        stats.kept_modules += len(filtered_modules)
        stats.dropped_modules += len(raw_modules) - len(filtered_modules)

        for module_name in filtered_modules:
            module_node = ("m", module_name)
            store.set_node_attrs(module_node, {"bipartite": 1})
            add_weighted_edge(store, key_node, module_node, policy=_EDGE_WEIGHT_POLICY)
    return stats


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
        return _empty_graph(directed=False)
    modules_reader = _scan_snapshot_reader(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="core.modules",
            snapshot_id=commit,
            columns=("module", "repo", "commit"),
        )
    )
    if modules_reader is None:
        return _empty_graph(directed=False)
    allowed_modules = _allowed_modules_from_reader(modules_reader, repo=repo, commit=commit)

    config_reader = _scan_snapshot_reader(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="analytics.config_values",
            snapshot_id=commit,
            columns=("key", "reference_modules", "repo", "commit"),
        )
    )
    if config_reader is None:
        return _empty_graph(directed=False)

    store = RxGraphStore.undirected()
    stats = _populate_config_graph(
        store,
        config_reader,
        repo=repo,
        commit=commit,
        allowed_modules=allowed_modules,
    )
    graph = rx_to_networkx(store.graph)
    log.info(
        "Config bipartite built: rows=%d empty_refs=%d allowed_modules=%d "
        "parsed_modules=%d kept_modules=%d dropped_modules=%d graph_nodes=%d edges=%d",
        stats.total_rows,
        stats.empty_refs,
        len(allowed_modules),
        stats.parsed_modules,
        stats.kept_modules,
        stats.dropped_modules,
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
        return _empty_graph(directed=False)
    edge_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.symbol_use_edges",
            snapshot_id=commit,
            columns=("def_path", "use_path"),
        )
    )
    if edge_table is None:
        return _empty_graph(directed=False)
    module_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="core.modules",
            snapshot_id=commit,
            columns=("path", "module", "repo", "commit"),
        )
    )
    if module_table is None:
        return _empty_graph(directed=False)
    module_by_path = _module_name_map(
        table_to_reader(_normalize_view_table(module_table)),
        repo=repo,
        commit=commit,
    )
    store = RxGraphStore.undirected()
    for def_path, use_path in iter_tuples_from_arrow_reader(
        table_to_reader(_normalize_view_table(edge_table))
    ):
        if def_path is None or use_path is None:
            continue
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        add_weighted_edge(store, use_module, def_module, policy=_EDGE_WEIGHT_POLICY)
    graph = rx_to_networkx(store.graph)
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
        return _empty_graph(directed=False)
    edge_table = scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root,
            table_key="graph.symbol_use_edges",
            snapshot_id=commit,
            columns=("def_goid_h128", "use_goid_h128"),
        )
    )
    if edge_table is None:
        return _empty_graph(directed=False)

    store = RxGraphStore.undirected()
    for def_goid, use_goid in iter_tuples_from_arrow_reader(
        table_to_reader(_normalize_view_table(edge_table))
    ):
        if def_goid is None or use_goid is None:
            continue
        left = normalize_decimal(def_goid)
        right = normalize_decimal(use_goid)
        if left is None or right is None or left == right:
            continue
        add_weighted_edge(store, left, right, policy=_EDGE_WEIGHT_POLICY)
    graph = rx_to_networkx(store.graph)
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)


__all__ = [
    "as_int",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "module_attrs_from_row",
    "normalize_decimal",
    "parse_reference_modules",
]
