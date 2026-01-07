"""Shared graph builders and edge weight helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.core.data_models.ids import as_int, normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Hashable


def _coerce_edge_weight(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class EdgeWeightPolicy:
    """Edge weight increment policy."""

    default: int = 1

    def next_weight(self, value: object | None) -> int:
        """Return the next weight when an edge is observed again.

        Returns
        -------
        int
            The incremented weight or the default when no prior weight exists.
        """
        parsed = _coerce_edge_weight(value)
        if parsed is None:
            return self.default
        return parsed + 1


def add_weighted_edge(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    *,
    policy: EdgeWeightPolicy | None = None,
) -> None:
    """Add or increment a weighted edge in the provided graph."""
    resolved = policy or EdgeWeightPolicy()
    if graph.has_edge(source, target):
        attrs = graph[source][target]
        attrs["weight"] = resolved.next_weight(attrs.get("weight"))
        return
    graph.add_edge(source, target, weight=resolved.default)


def add_call_graph_edges(
    graph: nx.DiGraph,
    rows: Iterable[Mapping[str, object]],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> None:
    """Append call graph edges from row mappings."""
    for row in rows:
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        add_weighted_edge(graph, caller, callee, policy=policy)


def add_call_graph_nodes(
    graph: nx.DiGraph,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Append call graph nodes from row mappings."""
    for row in rows:
        node_id = normalize_decimal_id(row.get("goid_h128"))
        if node_id is None or node_id in graph:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(node_id, **attrs)


def build_call_graph_from_rows(
    call_graph_edges: Iterable[Mapping[str, object]],
    call_graph_nodes: Iterable[Mapping[str, object]] | None = None,
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.DiGraph:
    """Build a call graph from scoped call graph edge/node rows.

    Returns
    -------
    nx.DiGraph
        Directed call graph populated from the provided rows.
    """
    graph = nx.DiGraph()
    add_call_graph_edges(graph, call_graph_edges, policy=policy)
    if call_graph_nodes is not None:
        add_call_graph_nodes(graph, call_graph_nodes)
    return graph


def add_import_edges(
    graph: nx.DiGraph,
    rows: Iterable[Mapping[str, object]],
    *,
    coerce_int: Callable[[object], int | None] = as_int,
    policy: EdgeWeightPolicy | None = None,
) -> dict[str, int]:
    """Append import graph edges and return inferred layer defaults.

    Returns
    -------
    dict[str, int]
        Fallback layer assignments keyed by module name.
    """
    fallback_layer_by_module: dict[str, int] = {}
    for row in rows:
        source_raw = row.get("src_module")
        target_raw = row.get("dst_module")
        if source_raw is None or target_raw is None:
            continue
        source = str(source_raw)
        target = str(target_raw)
        layer = coerce_int(row.get("module_layer"))
        if layer is not None:
            fallback_layer_by_module[source] = layer
        add_weighted_edge(graph, source, target, policy=policy)
    return fallback_layer_by_module


def add_import_module_rows(
    graph: nx.DiGraph,
    rows: Iterable[Mapping[str, object]] | None,
    *,
    fallback_layer_by_module: Mapping[str, int],
    coerce_int: Callable[[object], int | None] = as_int,
) -> None:
    """Append import module rows as nodes with attributes."""
    module_rows = list(rows or [])
    if module_rows:
        for row in module_rows:
            module = row.get("module")
            if module is None:
                continue
            module_name = str(module)
            attrs: dict[str, int] = {}
            scc_id = coerce_int(row.get("scc_id"))
            if scc_id is not None:
                attrs["scc_id"] = scc_id
            component_size = coerce_int(row.get("component_size"))
            if component_size is not None:
                attrs["component_size"] = component_size
            layer = coerce_int(row.get("layer"))
            if layer is not None:
                attrs["layer"] = layer
            if "layer" not in attrs and module_name in fallback_layer_by_module:
                attrs["layer"] = fallback_layer_by_module[module_name]
            graph.add_node(module_name, **attrs)
        return
    if fallback_layer_by_module:
        graph.add_nodes_from(
            [(module, {"layer": layer}) for module, layer in fallback_layer_by_module.items()]
        )


def build_import_graph_from_rows(
    import_graph_edges: Iterable[Mapping[str, object]],
    import_modules: Iterable[Mapping[str, object]] | None = None,
    *,
    coerce_int: Callable[[object], int | None] = as_int,
    policy: EdgeWeightPolicy | None = None,
) -> nx.DiGraph:
    """Build an import graph from scoped import edges and module rows.

    Returns
    -------
    nx.DiGraph
        Directed import graph populated from the provided rows.
    """
    graph = nx.DiGraph()
    fallback_layer_by_module = add_import_edges(
        graph,
        import_graph_edges,
        coerce_int=coerce_int,
        policy=policy,
    )
    add_import_module_rows(
        graph,
        import_modules,
        fallback_layer_by_module=fallback_layer_by_module,
        coerce_int=coerce_int,
    )
    return graph


def _map_path_to_module(value: object, module_by_path: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    return module_by_path.get(str(value))


def build_symbol_module_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.Graph:
    """Build an undirected weighted symbol-module graph from use edges.

    Returns
    -------
    nx.Graph
        Undirected symbol-module graph populated from the provided rows.
    """
    graph = nx.Graph()
    for record in symbol_use_edges:
        def_module = _map_path_to_module(record.get("def_path"), module_by_path)
        use_module = _map_path_to_module(record.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        add_weighted_edge(graph, use_module, def_module, policy=policy)
    return graph


def build_symbol_function_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.Graph:
    """Build an undirected weighted symbol-function graph from use edges.

    Returns
    -------
    nx.Graph
        Undirected symbol-function graph populated from the provided rows.
    """
    graph = nx.Graph()
    for record in symbol_use_edges:
        def_goid = normalize_decimal_id(record.get("def_goid_h128"))
        use_goid = normalize_decimal_id(record.get("use_goid_h128"))
        if def_goid is None or use_goid is None:
            continue
        if def_goid == use_goid:
            continue
        add_weighted_edge(graph, use_goid, def_goid, policy=policy)
    return graph


def build_symbol_module_edges(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate symbol use edges to module-level adjacency.

    Returns
    -------
    tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
        Modules, inbound adjacency, and outbound adjacency extracted from use edges.
    """
    modules: set[str] = set()
    inbound: dict[str, set[str]] = {}
    outbound: dict[str, set[str]] = {}
    for record in symbol_use_edges:
        def_module = _map_path_to_module(record.get("def_path"), module_by_path)
        use_module = _map_path_to_module(record.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        modules.update((use_module, def_module))
        inbound.setdefault(def_module, set()).add(use_module)
        outbound.setdefault(use_module, set()).add(def_module)
    return modules, inbound, outbound


__all__ = [
    "EdgeWeightPolicy",
    "add_call_graph_edges",
    "add_call_graph_nodes",
    "add_import_edges",
    "add_import_module_rows",
    "add_weighted_edge",
    "build_call_graph_from_rows",
    "build_import_graph_from_rows",
    "build_symbol_function_graph",
    "build_symbol_module_edges",
    "build_symbol_module_graph",
]
