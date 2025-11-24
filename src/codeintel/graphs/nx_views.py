"""Shared helpers to materialize DuckDB graphs as NetworkX views."""

from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import Decimal

import duckdb
import networkx as nx

from codeintel.storage.gateway import StorageGateway


def _normalize_decimal(value: object) -> int | None:
    """
    Normalize DuckDB DECIMAL(38,0) values to Python ints.

    Parameters
    ----------
    value : object
        Value from DuckDB rows representing a GOID.

    Returns
    -------
    int | None
        Integer representation or None when parsing fails.
    """
    result: int | None = None
    if value is None:
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, Decimal):
        result = int(value)
    elif isinstance(value, (bytes, bytearray)):
        try:
            result = int(value.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            result = None
    else:
        try:
            result = int(str(value))
        except (TypeError, ValueError):
            result = None
    return result


def _module_attrs_from_row(
    module: object,
    scc_id: object,
    component_size: object,
    layer: object,
    cycle_group: object,
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
    cycle_group :
        Cycle grouping id retained for backwards compatibility.

    Returns
    -------
    tuple[str, dict[str, int]]
        Normalized module name and attribute dictionary.
    """
    module_name = str(module)
    attrs: dict[str, int] = {}
    if scc_id is not None:
        attrs["scc_id"] = int(scc_id)
    if component_size is not None:
        attrs["component_size"] = int(component_size)
    if layer is not None:
        attrs["layer"] = int(layer)
    if cycle_group is not None:
        attrs["cycle_group"] = int(cycle_group)
    return module_name, attrs


def load_call_graph(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> nx.DiGraph:
    """
    Build a call graph `DiGraph` of caller -> callee edges.

    Nodes are GOID integers; parallel edges are aggregated via `weight`.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.

    Returns
    -------
    nx.DiGraph
        Directed call graph with weighted edges and isolated nodes preserved.
    """
    con = gateway.con
    rows: Iterable[tuple[object, object | None]] = con.execute(
        """
        SELECT caller_goid_h128, callee_goid_h128
        FROM graph.call_graph_edges
        WHERE callee_goid_h128 IS NOT NULL
          AND repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    graph = nx.DiGraph()
    for caller_raw, callee_raw in rows:
        caller = _normalize_decimal(caller_raw)
        callee = _normalize_decimal(callee_raw)
        if caller is None or callee is None:
            continue
        if graph.has_edge(caller, callee):
            graph[caller][callee]["weight"] += 1
        else:
            graph.add_edge(caller, callee, weight=1)

    # Ensure isolated nodes are present
    node_rows = con.execute(
        """
        SELECT goid_h128
        FROM graph.call_graph_nodes
        """
    ).fetchall()
    for (node_raw,) in node_rows:
        node = _normalize_decimal(node_raw)
        if node is not None and node not in graph:
            graph.add_node(node)

    return graph


def load_import_graph(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> nx.DiGraph:
    """
    Build a directed import graph `DiGraph` of module -> module edges.

    Edge weights represent aggregated edge counts when multiple edges exist.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.

    Returns
    -------
    nx.DiGraph
        Directed import graph with weights capturing edge multiplicity.
    """
    con = gateway.con
    edge_rows = con.execute(
        """
        SELECT src_module, dst_module, module_layer
        FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    graph = nx.DiGraph()
    fallback_layer_by_module: dict[str, int] = {}
    for src, dst, layer in edge_rows:
        if src is None or dst is None:
            continue
        source = str(src)
        target = str(dst)
        if layer is not None:
            fallback_layer_by_module[source] = int(layer)
        weight = graph.get_edge_data(source, target, {}).get("weight", 0) + 1
        graph.add_edge(source, target, weight=weight)

    try:
        module_rows = con.execute(
            """
            SELECT module, scc_id, component_size, layer, cycle_group
            FROM graph.import_modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except duckdb.Error:
        module_rows = []
    if module_rows:
        for module_row in module_rows:
            module_name, attrs = _module_attrs_from_row(*module_row)
            graph.add_node(module_name, **attrs)
    elif fallback_layer_by_module:
        graph.add_nodes_from(
            [(module, {"layer": layer}) for module, layer in fallback_layer_by_module.items()]
        )
    return graph


def load_test_function_bipartite(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> nx.Graph:
    """
    Build a bipartite graph of tests <-> functions from coverage edges.

    Test nodes are keyed as ("t", test_id); function nodes as ("f", goid).
    Edge weight is derived from coverage_ratio when present.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.

    Returns
    -------
    nx.Graph
        Undirected bipartite graph with weighted coverage edges.
    """
    con = gateway.con
    rows: Iterable[tuple[str, object, float | None]] = con.execute(
        """
        SELECT test_id, function_goid_h128, COALESCE(coverage_ratio, 0.0)
        FROM analytics.test_coverage_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    graph = nx.Graph()
    for test_id, goid_raw, coverage_ratio in rows:
        goid = _normalize_decimal(goid_raw)
        if test_id is None or goid is None:
            continue
        test_node = ("t", str(test_id))
        func_node = ("f", goid)
        if not graph.has_node(test_node):
            graph.add_node(test_node, bipartite=0)
        if not graph.has_node(func_node):
            graph.add_node(func_node, bipartite=1)
        weight = float(coverage_ratio or 0.0)
        if graph.has_edge(test_node, func_node):
            graph[test_node][func_node]["weight"] += weight
        else:
            graph.add_edge(test_node, func_node, weight=weight)
    return graph


def _parse_reference_modules(ref_modules: object, allowed_modules: set[str]) -> list[str]:
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
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> nx.Graph:
    """
    Build a bipartite graph of config keys <-> modules.

    Keys are ("c", key); modules are ("m", module). Edge weight equals one per
    reference occurrence.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.

    Returns
    -------
    nx.Graph
        Undirected bipartite graph for configuration references.
    """
    con = gateway.con
    allowed_modules = {
        str(mod)
        for (mod,) in con.execute(
            "SELECT module FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit]
        ).fetchall()
    }

    rows: Iterable[tuple[object, object]] = con.execute(
        """
        SELECT key, reference_modules
        FROM analytics.config_values
        """,
    ).fetchall()

    graph = nx.Graph()
    for key, ref_modules in rows:
        if key is None or ref_modules is None:
            continue
        key_node = ("c", str(key))
        if not graph.has_node(key_node):
            graph.add_node(key_node, bipartite=0)

        modules = _parse_reference_modules(ref_modules, allowed_modules)
        for module_name in modules:
            module_node = ("m", module_name)
            if not graph.has_node(module_node):
                graph.add_node(module_node, bipartite=1)
            if graph.has_edge(key_node, module_node):
                graph[key_node][module_node]["weight"] += 1
            else:
                graph.add_edge(key_node, module_node, weight=1)
    return graph


def load_symbol_module_graph(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> nx.Graph:
    """
    Build an undirected weighted graph of module-level symbol coupling.

    Edge weights count shared symbol def/use pairs between modules.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.

    Returns
    -------
    nx.Graph
        Undirected graph where weights reflect shared symbol relations.
    """
    con = gateway.con
    rows = con.execute(
        """
        SELECT m_use.module AS use_module, m_def.module AS def_module
        FROM graph.symbol_use_edges su
        LEFT JOIN core.modules m_def ON m_def.path = su.def_path
        LEFT JOIN core.modules m_use ON m_use.path = su.use_path
        WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
          AND (m_def.repo = ? OR m_def.repo IS NULL)
          AND (m_use.repo = ? OR m_use.repo IS NULL)
          AND (m_def.commit = ? OR m_def.commit IS NULL)
          AND (m_use.commit = ? OR m_use.commit IS NULL)
        """,
        [repo, repo, commit, commit],
    ).fetchall()

    graph = nx.Graph()
    for use_module, def_module in rows:
        if use_module is None or def_module is None:
            continue
        left = str(use_module)
        right = str(def_module)
        if left == right:
            continue
        if graph.has_edge(left, right):
            graph[left][right]["weight"] += 1
        else:
            graph.add_edge(left, right, weight=1)
    return graph


def load_symbol_function_graph(
    gateway: StorageGateway,
    _repo: str,
    _commit: str,
) -> nx.Graph:
    """
    Build an undirected weighted graph of function-level symbol coupling (GOIDs).

    Edge weights count shared symbol def/use pairs between functions when available.

    Parameters
    ----------
    gateway :
        Gateway providing the DuckDB connection scoped to the target repository.

    Returns
    -------
    nx.Graph
        Undirected graph linking functions by shared symbol usage.
    """
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT su.def_goid_h128, su.use_goid_h128
            FROM graph.symbol_use_edges su
            WHERE su.def_goid_h128 IS NOT NULL
              AND su.use_goid_h128 IS NOT NULL
            """,
        ).fetchall()
    except duckdb.Error:
        return nx.Graph()

    graph = nx.Graph()
    for def_goid, use_goid in rows:
        if def_goid is None or use_goid is None:
            continue
        left = _normalize_decimal(def_goid)
        right = _normalize_decimal(use_goid)
        if left is None or right is None or left == right:
            continue
        if graph.has_edge(left, right):
            graph[left][right]["weight"] += 1
        else:
            graph.add_edge(left, right, weight=1)
    return graph
