"""Shared helpers to materialize Parquet-backed graphs as rustworkx stores.

This module provides functions to load various graph types from
Parquet datasets into rustworkx graph stores. View-registry
fallthrough is intentionally disallowed in this layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.graphs.builders import add_weighted_edge
from codeintel.build.graphs.engine.datasets import GraphViewFactory, GraphViewScanOptions
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.policies import weight_policy_for_kind
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.data_models.ids import as_int
from codeintel.core.data_models.ids import normalize_decimal_id as normalize_decimal

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


def _ensure_dataset_root(dataset_root: Path | None, table_key: str) -> Path | None:
    if dataset_root is None:
        log.warning("Dataset root not configured; cannot load %s", table_key)
        return None
    return dataset_root


def _view_factory(
    dataset_root: Path,
    *,
    repo: str | None,
    commit: str,
) -> GraphViewFactory:
    return GraphViewFactory.for_snapshot(dataset_root, repo=repo, commit=commit)


def _column_index(names: list[str], column: str) -> int | None:
    try:
        return names.index(column)
    except ValueError:
        return None


def _iter_scoped_rows(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> Iterable[tuple[object, ...]]:
    names = list(reader.schema.names)
    repo_idx = _column_index(names, "repo")
    commit_idx = _column_index(names, "commit")
    repo = factory.scan_context.repo
    commit = factory.scan_context.commit
    for row in factory.iter_tuples(reader):
        if repo_idx is not None and repo is not None:
            row_repo = row[repo_idx]
            if row_repo is not None and str(row_repo) != repo:
                continue
        if commit_idx is not None and commit is not None:
            row_commit = row[commit_idx]
            if row_commit is not None and str(row_commit) != commit:
                continue
        yield row


def _empty_graph(*, directed: bool, kind: GraphKind) -> RxGraphStore:
    policy = weight_policy_for_kind(kind)
    return (
        RxGraphStore.directed(weight_policy=policy)
        if directed
        else RxGraphStore.undirected(weight_policy=policy)
    )


def _module_name_map(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    specificity_by_path: dict[str, int] = {}
    repo = factory.scan_context.repo
    commit = factory.scan_context.commit
    for path, module, row_repo, row_commit in factory.iter_tuples(reader):
        if path is None or module is None:
            continue
        if repo is not None and row_repo is not None and str(row_repo) != repo:
            continue
        if commit is not None and row_commit is not None and str(row_commit) != commit:
            continue
        specificity = int(row_repo is not None) + int(row_commit is not None)
        key = str(path)
        if specificity < specificity_by_path.get(key, -1):
            continue
        module_by_path[key] = str(module)
        specificity_by_path[key] = specificity
    return module_by_path


def _add_call_edges(
    store: RxGraphStore,
    rows: Iterable[tuple[object, ...]],
) -> None:
    for caller_raw, callee_raw in rows:
        caller = normalize_decimal(caller_raw)
        callee = normalize_decimal(callee_raw)
        if caller is None or callee is None:
            continue
        add_weighted_edge(store, caller, callee)


def _add_call_nodes(
    store: RxGraphStore,
    rows: Iterable[tuple[object, ...]],
) -> None:
    for node_raw, kind in rows:
        node = normalize_decimal(node_raw)
        if node is None:
            continue
        attrs: dict[str, object] = {}
        if kind is not None:
            attrs["kind"] = str(kind)
        store.set_node_attrs(node, attrs)


def _maybe_to_gpu_graph(store: RxGraphStore, *, use_gpu: bool) -> RxGraphStore:
    """
    No-op for rustworkx-backed execution (CPU-only).

    Parameters
    ----------
    store : RxGraphStore
        Graph store to optionally prepare for GPU execution.
    use_gpu : bool
        Whether GPU execution was requested.

    Returns
    -------
    RxGraphStore
        The original graph store.
    """
    if use_gpu:
        log.debug("GPU backend requested; rustworkx execution is CPU-only.")
    return store


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
) -> RxGraphStore:
    """
    Build a call graph store of caller -> callee edges.

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
    RxGraphStore
        Directed call graph store with weighted edges and isolated nodes preserved.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.call_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.call_graph_edges",
        columns=("caller_goid_h128", "callee_goid_h128"),
    )
    if edge_reader is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)

    store = RxGraphStore.directed(weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH))
    _add_call_edges(store, factory.iter_tuples(edge_reader))

    node_reader = factory.load_reader(
        table_key="graph.call_graph_nodes",
        columns=("goid_h128", "kind"),
    )
    if node_reader is not None:
        _add_call_nodes(store, factory.iter_tuples(node_reader))

    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_import_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a directed import graph store of module -> module edges.

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
    RxGraphStore
        Directed import graph store with weights capturing edge multiplicity.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.import_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.import_graph_edges",
        columns=("src_module", "dst_module", "module_layer"),
    )
    if edge_reader is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)

    store = RxGraphStore.directed(weight_policy=weight_policy_for_kind(GraphKind.IMPORT_GRAPH))
    fallback_layer_by_module: dict[str, int] = {}
    for src, dst, layer in factory.iter_tuples(edge_reader):
        if src is None or dst is None:
            continue
        source = str(src)
        target = str(dst)
        layer_value = as_int(layer)
        if layer_value is not None:
            fallback_layer_by_module[source] = layer_value
        add_weighted_edge(store, source, target)

    module_reader = factory.load_reader(
        table_key="graph.import_modules",
        columns=("module", "scc_id", "component_size", "layer"),
    )
    if module_reader is not None:
        for module_row in factory.iter_tuples(module_reader):
            module_name, attrs = module_attrs_from_row(*module_row)
            store.set_node_attrs(module_name, attrs)
    elif fallback_layer_by_module:
        for module, layer in fallback_layer_by_module.items():
            store.set_node_attrs(module, {"layer": layer})
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def parse_reference_modules(ref_modules: object, allowed_modules: set[str]) -> list[str]:
    """Normalize reference modules input into a filtered list.

    Returns
    -------
    list[str]
        Allowed module names parsed from input.
    """
    modules: list[str] = []
    if isinstance(ref_modules, Mapping):
        ref_modules = ref_modules.get("reference_modules")
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
    factory: GraphViewFactory,
    modules_reader: pa.RecordBatchReader,
) -> set[str]:
    names = list(modules_reader.schema.names)
    module_idx = _column_index(names, "module")
    if module_idx is None:
        return set()
    allowed: set[str] = set()
    for row in _iter_scoped_rows(factory, modules_reader):
        value = row[module_idx]
        if value is None:
            continue
        allowed.add(str(value))
    return allowed


def _populate_config_graph(
    store: RxGraphStore,
    factory: GraphViewFactory,
    config_reader: pa.RecordBatchReader,
    *,
    allowed_modules: set[str],
) -> ConfigGraphStats:
    stats = ConfigGraphStats()
    names = list(config_reader.schema.names)
    key_idx = _column_index(names, "key")
    extras_idx = _column_index(names, "extras")
    if key_idx is None or extras_idx is None:
        return stats
    for row in _iter_scoped_rows(factory, config_reader):
        stats.total_rows += 1
        key = row[key_idx]
        ref_modules = row[extras_idx]
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
            add_weighted_edge(store, key_node, module_node)
    return stats


def load_config_module_bipartite(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a bipartite graph store of config keys <-> modules.

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
    RxGraphStore
        Undirected bipartite graph store for configuration references.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "analytics.config_values")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    modules_reader = factory.load_reader(
        table_key="core.modules",
        columns=("module", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if modules_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    allowed_modules = _allowed_modules_from_reader(factory, modules_reader)

    config_reader = factory.load_reader(
        table_key="analytics.config_values",
        columns=("key", "extras", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if config_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)

    store = RxGraphStore.undirected(
        weight_policy=weight_policy_for_kind(GraphKind.CONFIG_MODULE_BIPARTITE)
    )
    stats = _populate_config_graph(
        store,
        factory,
        config_reader,
        allowed_modules=allowed_modules,
    )
    graph = store.graph
    log.info(
        "Config bipartite built: rows=%d empty_refs=%d allowed_modules=%d "
        "parsed_modules=%d kept_modules=%d dropped_modules=%d graph_nodes=%d edges=%d",
        stats.total_rows,
        stats.empty_refs,
        len(allowed_modules),
        stats.parsed_modules,
        stats.kept_modules,
        stats.dropped_modules,
        graph.num_nodes(),
        graph.num_edges(),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_module_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of module-level symbol coupling.

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
    RxGraphStore
        Undirected graph store where weights reflect shared symbol relations.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.symbol_use_edges",
        columns=("def_path", "use_path"),
    )
    if edge_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    module_reader = factory.load_reader(
        table_key="core.modules",
        columns=("path", "module", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if module_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    module_by_path = _module_name_map(factory, module_reader)
    store = RxGraphStore.undirected(
        weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_MODULE_GRAPH)
    )
    for def_path, use_path in factory.iter_tuples(edge_reader):
        if def_path is None or use_path is None:
            continue
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        add_weighted_edge(store, use_module, def_module)
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_function_graph(
    dataset_root: Path | None,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of function-level symbol coupling (GOIDs).

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
    RxGraphStore
        Undirected graph store linking functions by shared symbol usage.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)
    factory = _view_factory(dataset_root, repo=None, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.symbol_use_edges",
        columns=("def_goid_h128", "use_goid_h128"),
    )
    if edge_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)

    store = RxGraphStore.undirected(
        weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_FUNCTION_GRAPH)
    )
    for def_goid, use_goid in factory.iter_tuples(edge_reader):
        if def_goid is None or use_goid is None:
            continue
        left = normalize_decimal(def_goid)
        right = normalize_decimal(use_goid)
        if left is None or right is None or left == right:
            continue
        add_weighted_edge(store, left, right)
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


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
