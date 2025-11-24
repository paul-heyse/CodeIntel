"""Compute CFG/DFG-derived metrics using NetworkX views."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast

import duckdb
import networkx as nx
from networkx.exception import NetworkXError, NetworkXNoPath, PowerIterationFailedConvergence

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import _normalize_decimal
from codeintel.storage.gateway import StorageGateway


def _to_decimal(value: int) -> Decimal:
    return Decimal(value)


MAX_CFG_CENTRALITY_SAMPLE = 100
MAX_CFG_EIGEN_SAMPLE = 200
MAX_DFG_CENTRALITY_SAMPLE = 100
MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50


def _bounded_simple_path_count(graph: nx.DiGraph, sources: set[int], targets: set[int]) -> int:
    """
    Count simple paths between sources and targets with hard limits.

    Parameters
    ----------
    graph :
        Directed graph to traverse.
    sources :
        Set of source node identifiers.
    targets :
        Set of target node identifiers.

    Returns
    -------
    int
        Number of simple paths discovered up to MAX_SIMPLE_PATHS.
    """
    count = 0
    for source in sources:
        for target in targets:
            if count >= MAX_SIMPLE_PATHS:
                return count
            try:
                paths = nx.all_simple_paths(
                    graph, source=source, target=target, cutoff=MAX_PATH_CUTOFF
                )
            except NetworkXError:
                continue
            for _ in paths:
                count += 1
                if count >= MAX_SIMPLE_PATHS:
                    return count
    return count


@dataclass(frozen=True)
class CfgFnContext:
    """Context bundle for computing CFG function and block metric rows."""

    repo: str
    commit: str
    fn_goid: int
    rel_path: str
    module: str | None
    qualname: str | None
    graph: nx.DiGraph
    entry_idx: int
    exit_idx: int
    sccs: list[set[int]]
    now: datetime


@dataclass(frozen=True)
class CfgInputs:
    """Shared CFG inputs for per-function metric derivation."""

    repo: str
    commit: str
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]]
    edges_by_fn: dict[int, list[tuple[int, int, str]]]
    now: datetime


@dataclass(frozen=True)
class CfgFnRows:
    """Container for per-function CFG rows."""

    fn_row: tuple[object, ...]
    ext_row: tuple[object, ...]
    block_rows: list[tuple[object, ...]]


@dataclass(frozen=True)
class DfgFnContext:
    """Context bundle for computing DFG function and block metric rows."""

    repo: str
    commit: str
    fn_goid: int
    rel_path: str
    module: str | None
    qualname: str | None
    graph: nx.DiGraph
    phi_edges: int
    symbol_count: int
    components_count: int
    sccs: list[set[int]]
    has_cycles: bool
    longest_chain: int
    avg_spl: float
    dfg_in_deg: dict[int, int]
    dfg_out_deg: dict[int, int]
    dfg_phi_in: dict[int, int]
    dfg_phi_out: dict[int, int]
    branchy_fraction: float
    bc: dict[int, float]
    eig: dict[int, float]
    now: datetime


@dataclass(frozen=True)
class DfgInputs:
    """Input payload for building a DFG metrics context."""

    fn_goid: int
    meta: tuple[str, str | None, str | None]
    edges: list[tuple[int, int, str, str, bool, str]]
    repo: str
    commit: str
    now: datetime


def _degree_dict(
    graph: nx.DiGraph,
    *,
    direction: Literal["in", "out"],
    weight: str | None = None,
) -> dict[int, int]:
    """
    Materialize degree counts into a concrete mapping for type safety.

    Using explicit dicts avoids DegreeView typing ambiguity in strict checkers.

    Returns
    -------
    dict[int, int]
        Mapping of node -> degree.
    """
    raw_pairs = (
        graph.in_degree(weight=weight) if direction == "in" else graph.out_degree(weight=weight)
    )
    pairs = cast("Iterable[tuple[int, int | float]]", raw_pairs)
    return {int(node): int(deg) for node, deg in pairs}


def _load_cfg_blocks(
    gateway: StorageGateway, _repo: str, _commit: str
) -> tuple[dict[int, list[tuple[int, str, int, int]]], dict[int, list[tuple[int, int, str]]]]:
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    edges_by_fn: dict[int, list[tuple[int, int, str]]] = defaultdict(list)

    try:
        block_rows: Iterable[tuple[int, int, str, int, int]] = gateway.con.execute(
            """
            SELECT function_goid_h128::BIGINT, block_idx, kind, in_degree, out_degree
            FROM graph.cfg_blocks
            """
        ).fetchall()
    except duckdb.Error:
        return blocks_by_fn, edges_by_fn
    for fn, idx, kind, in_deg, out_deg in block_rows:
        blocks_by_fn[int(fn)].append((int(idx), str(kind), int(in_deg), int(out_deg)))

    try:
        edge_rows: Iterable[tuple[int, int, int, str]] = gateway.con.execute(
            """
            SELECT function_goid_h128::BIGINT, src_block_id, dst_block_id, edge_kind
            FROM graph.cfg_edges
            """
        ).fetchall()
    except duckdb.Error:
        return blocks_by_fn, edges_by_fn
    for fn, src_id, dst_id, edge_type in edge_rows:
        src_idx = _parse_block_idx(src_id) if src_id is not None else None
        dst_idx = _parse_block_idx(dst_id) if dst_id is not None else None
        if src_idx is None or dst_idx is None:
            continue
        edges_by_fn[int(fn)].append((src_idx, dst_idx, str(edge_type)))

    return blocks_by_fn, edges_by_fn


def _parse_block_idx(block_id: str | int | None) -> int | None:
    """
    Extract the integer block index from a block identifier.

    Parameters
    ----------
    block_id : str | None
        Block identifier shaped like "<goid>:block<idx>".

    Returns
    -------
    int | None
        Parsed block index or None when parsing fails.
    """
    if block_id is None:
        return None
    block_text = str(block_id)
    if "block" not in block_text:
        return None
    try:
        return int(block_text.rsplit("block", 1)[-1])
    except ValueError:
        return None


def _load_dfg_edges(
    gateway: StorageGateway, _repo: str, _commit: str
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    try:
        rows: Iterable[tuple[int, int, int, str, str, bool, str]] = gateway.con.execute(
            """
            SELECT function_goid_h128::BIGINT, src_block_id, dst_block_id,
                   src_var, dst_var, via_phi, use_kind
            FROM graph.dfg_edges
            """
        ).fetchall()
    except duckdb.Error:
        return edges_by_fn
    for fn, src_id, dst_id, src_sym, dst_sym, via_phi, use_kind in rows:
        src_idx = _parse_block_idx(src_id)
        dst_idx = _parse_block_idx(dst_id)
        if src_idx is None or dst_idx is None:
            continue
        edges_by_fn[int(fn)].append(
            (src_idx, dst_idx, str(src_sym), str(dst_sym), bool(via_phi), str(use_kind))
        )
    return edges_by_fn


def _function_metadata(
    gateway: StorageGateway, repo: str, commit: str
) -> dict[int, tuple[str, str | None, str | None]]:
    rows: Iterable[tuple[object, str, str | None, str | None]] = gateway.con.execute(
        """
        SELECT g.goid_h128,
               g.rel_path,
               m.module,
               g.qualname
        FROM core.goids g
        LEFT JOIN core.modules m
          ON m.path = g.rel_path
        WHERE g.repo = ? AND g.commit = ?
          AND g.kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetchall()
    result: dict[int, tuple[str, str | None, str | None]] = {}
    for goid_raw, rel_path, module, qualname in rows:
        goid = _normalize_decimal(goid_raw)
        if goid is None:
            continue
        result[int(goid)] = (rel_path, module, qualname)
    return result


def _branching_stats(graph: nx.DiGraph) -> tuple[float, int, float]:
    in_degrees = _degree_dict(graph, direction="in")
    out_degrees_map = _degree_dict(graph, direction="out")
    out_degrees = [deg for deg in out_degrees_map.values() if deg > 0]
    branching_mean = (sum(out_degrees) / len(out_degrees)) if out_degrees else 0.0
    branching_max = max(out_degrees) if out_degrees else 0
    linear_blocks = [
        n for n in graph.nodes if in_degrees.get(n, 0) == 1 and out_degrees_map.get(n, 0) == 1
    ]
    linear_fraction = (
        len(linear_blocks) / graph.number_of_nodes() if graph.number_of_nodes() else 0.0
    )
    return branching_mean, branching_max, linear_fraction


def _loop_stats(sccs: list[set[int]]) -> tuple[int, int]:
    loop_sccs = [comp for comp in sccs if len(comp) > 1]
    loop_count = len(loop_sccs)
    loop_max = max((len(comp) for comp in loop_sccs), default=0)
    return loop_count, loop_max


def _compute_cfg_centralities(
    graph: nx.DiGraph, entry_idx: int
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, int], dict[int, int]]:
    dom_depth: dict[int, int] = {}
    dom_frontier_sizes: dict[int, int] = {}
    dom_tree_height: dict[str, int | None] = {"height": None}

    try:
        idom = nx.immediate_dominators(graph, entry_idx)
        for node in graph.nodes:
            depth = 0
            cur = node
            while cur != entry_idx and cur in idom:
                cur = idom[cur]
                depth += 1
            dom_depth[node] = depth
        dom_tree_height["height"] = max(dom_depth.values()) if dom_depth else None
        frontier = nx.dominance_frontiers(graph, entry_idx)
        dom_frontier_sizes = {node: len(frontier.get(node, ())) for node in graph.nodes}
    except NetworkXError:
        dom_frontier_sizes = {}
        dom_tree_height["height"] = None

    try:
        bc = nx.betweenness_centrality(
            graph,
            k=min(MAX_CFG_CENTRALITY_SAMPLE, graph.number_of_nodes())
            if graph.number_of_nodes() > MAX_CFG_CENTRALITY_SAMPLE
            else None,
        )
    except NetworkXError:
        bc = dict.fromkeys(graph.nodes, 0.0)

    closeness = nx.closeness_centrality(graph) if graph.number_of_nodes() > 0 else {}
    try:
        eig = (
            nx.eigenvector_centrality(graph.to_undirected(), max_iter=MAX_CFG_EIGEN_SAMPLE)
            if graph.number_of_nodes() > 0
            else {}
        )
    except (NetworkXError, PowerIterationFailedConvergence):
        eig = dict.fromkeys(graph.nodes, 0.0)

    return bc, closeness, eig, dom_depth, dom_frontier_sizes


def _cfg_longest_path(graph: nx.DiGraph, entry_idx: int, *, is_dag: bool) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    if is_dag:
        try:
            reachable = nx.descendants(graph, entry_idx) | {entry_idx}
            subgraph = graph.subgraph(reachable).copy()
            return nx.dag_longest_path_length(nx.DiGraph(subgraph))
        except NetworkXNoPath:
            return 0
    condensation = nx.condensation(graph)
    return nx.dag_longest_path_length(condensation)


def _cfg_avg_shortest_path(graph: nx.DiGraph, entry_idx: int) -> float:
    try:
        lengths = nx.single_source_shortest_path_length(graph, entry_idx)
        return sum(lengths.values()) / len(lengths) if lengths else 0.0
    except NetworkXError:
        return 0.0


def _loop_nodes(sccs: list[set[int]]) -> set[int]:
    return set().union(*[comp for comp in sccs if len(comp) > 1]) if sccs else set()


def _build_dfg_graph(
    edges: list[tuple[int, int, str, str, bool, str]],
) -> tuple[nx.DiGraph, int, int]:
    graph: nx.DiGraph = nx.DiGraph()
    phi_edges = 0
    symbols: set[str] = set()
    for src, dst, src_sym, dst_sym, via_phi, use_kind in edges:
        graph.add_edge(
            src,
            dst,
            src_symbol=src_sym,
            dst_symbol=dst_sym,
            via_phi=via_phi,
            use_kind=use_kind,
        )
        symbols.add(src_sym)
        symbols.add(dst_sym)
        if via_phi:
            phi_edges += 1
    return graph, phi_edges, len(symbols)


def _dfg_path_lengths(graph: nx.DiGraph) -> tuple[int, float]:
    if graph.number_of_nodes() == 0:
        return 0, 0.0
    if nx.is_directed_acyclic_graph(graph):
        longest_chain = nx.dag_longest_path_length(graph)
    else:
        longest_chain = nx.dag_longest_path_length(nx.condensation(graph))

    all_lengths: list[int] = []
    for node in graph.nodes:
        lengths = nx.single_source_shortest_path_length(graph, node)
        all_lengths.extend(lengths.values())
    avg_spl = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
    return longest_chain, avg_spl


def _dfg_centralities(graph: nx.DiGraph) -> tuple[dict[int, float], dict[int, float]]:
    if graph.number_of_nodes() == 0:
        return {}, {}
    try:
        bc = nx.betweenness_centrality(
            graph,
            k=min(MAX_DFG_CENTRALITY_SAMPLE, graph.number_of_nodes())
            if graph.number_of_nodes() > MAX_DFG_CENTRALITY_SAMPLE
            else None,
        )
    except NetworkXError:
        bc = dict.fromkeys(graph.nodes, 0.0)
    try:
        eig = nx.eigenvector_centrality(graph.to_undirected(), max_iter=MAX_CFG_EIGEN_SAMPLE)
    except (NetworkXError, PowerIterationFailedConvergence):
        eig = dict.fromkeys(graph.nodes, 0.0)
    return bc, eig


def _build_dfg_context(inputs: DfgInputs) -> DfgFnContext | None:
    meta = inputs.meta
    if not inputs.edges:
        return None

    graph, phi_edges, symbol_count = _build_dfg_graph(inputs.edges)
    dfg_in_deg = _degree_dict(graph, direction="in")
    dfg_out_deg = _degree_dict(graph, direction="out")
    dfg_phi_in = dict.fromkeys(graph.nodes, 0)
    dfg_phi_out = dict.fromkeys(graph.nodes, 0)
    for src, dst, data in graph.edges(data=True):
        if data.get("via_phi"):
            dfg_phi_out[src] += 1
            dfg_phi_in[dst] += 1

    components_count = sum(1 for _ in nx.weakly_connected_components(graph))
    sccs = list(nx.strongly_connected_components(graph))
    has_cycles = any(len(comp) > 1 for comp in sccs)
    path_lengths = _dfg_path_lengths(graph)
    centralities = _dfg_centralities(graph)

    return DfgFnContext(
        repo=inputs.repo,
        commit=inputs.commit,
        fn_goid=inputs.fn_goid,
        rel_path=meta[0],
        module=meta[1],
        qualname=meta[2],
        graph=graph,
        phi_edges=phi_edges,
        symbol_count=symbol_count,
        components_count=components_count,
        sccs=sccs,
        has_cycles=has_cycles,
        longest_chain=path_lengths[0],
        avg_spl=path_lengths[1],
        dfg_in_deg=dfg_in_deg,
        dfg_out_deg=dfg_out_deg,
        dfg_phi_in=dfg_phi_in,
        dfg_phi_out=dfg_phi_out,
        branchy_fraction=(
            sum(1 for val in dfg_out_deg.values() if val > 1) / len(dfg_out_deg.values())
            if dfg_out_deg
            else 0.0
        ),
        bc=centralities[0],
        eig=centralities[1],
        now=inputs.now,
    )


def _dfg_fn_row(ctx: DfgFnContext) -> tuple[object, ...]:
    in_degs = list(ctx.dfg_in_deg.values())
    out_degs = list(ctx.dfg_out_deg.values())
    return (
        _to_decimal(ctx.fn_goid),
        ctx.repo,
        ctx.commit,
        ctx.rel_path,
        ctx.module,
        ctx.qualname,
        ctx.graph.number_of_nodes(),
        ctx.graph.number_of_edges(),
        ctx.phi_edges,
        ctx.symbol_count,
        ctx.components_count,
        len(ctx.sccs),
        ctx.has_cycles,
        ctx.longest_chain,
        ctx.avg_spl,
        (sum(in_degs) / len(in_degs)) if in_degs else 0.0,
        (sum(out_degs) / len(out_degs)) if out_degs else 0.0,
        max(in_degs) if in_degs else 0,
        max(out_degs) if out_degs else 0,
        ctx.branchy_fraction,
        max(ctx.bc.values()) if ctx.bc else 0.0,
        (sum(ctx.bc.values()) / len(ctx.bc)) if ctx.bc else 0.0,
        max(ctx.eig.values()) if ctx.eig else 0.0,
        ctx.now,
        1,
    )


def _dfg_block_rows(ctx: DfgFnContext) -> list[tuple[object, ...]]:
    loop_nodes = {node for comp in ctx.sccs if len(comp) > 1 for node in comp}
    return [
        (
            _to_decimal(ctx.fn_goid),
            ctx.repo,
            ctx.commit,
            node,
            ctx.dfg_in_deg.get(node, 0),
            ctx.dfg_out_deg.get(node, 0),
            ctx.dfg_phi_in.get(node, 0),
            ctx.dfg_phi_out.get(node, 0),
            ctx.bc.get(node, 0.0),
            None,
            ctx.eig.get(node, 0.0),
            node in loop_nodes,
            False,
            ctx.now,
            1,
        )
        for node in ctx.graph.nodes
    ]


def _dfg_ext_row(ctx: DfgFnContext) -> tuple[object, ...]:
    edge_kinds = Counter(data.get("use_kind") for _, _, data in ctx.graph.edges(data=True))
    data_flow_edges = edge_kinds.get("data-flow", 0)
    intra_block_edges = edge_kinds.get("intra-block", 0)
    phi_edges = sum(1 for _, _, data in ctx.graph.edges(data=True) if data.get("via_phi"))
    total_edges = ctx.graph.number_of_edges() or 1
    phi_ratio = phi_edges / total_edges
    other_kinds = sum(
        count
        for kind, count in edge_kinds.items()
        if kind not in {"data-flow", "intra-block", "phi"}
    )

    sources = {node for node in ctx.graph.nodes if ctx.graph.in_degree(node) == 0}
    sinks = {node for node in ctx.graph.nodes if ctx.graph.out_degree(node) == 0}
    simple_paths = _bounded_simple_path_count(ctx.graph, sources, sinks)

    return (
        _to_decimal(ctx.fn_goid),
        ctx.repo,
        ctx.commit,
        data_flow_edges,
        intra_block_edges,
        edge_kinds.get("phi", 0),
        data_flow_edges,
        intra_block_edges,
        other_kinds,
        phi_ratio,
        simple_paths,
        ctx.now,
        1,
    )


def _build_cfg_graph(
    blocks: list[tuple[int, str, int, int]], edges: list[tuple[int, int, str]]
) -> tuple[nx.DiGraph, int, int]:
    graph = nx.DiGraph()
    entry_idx = None
    exit_idx = None
    out_deg_map: dict[int, int] = {}
    for idx, kind, in_deg, out_deg in blocks:
        graph.add_node(idx, kind=kind, in_degree=in_deg, out_degree=out_deg)
        if kind == "entry":
            entry_idx = idx
        if kind == "exit":
            exit_idx = idx
        out_deg_map[idx] = out_deg
    for src, dst, edge_type in edges:
        graph.add_edge(src, dst, edge_type=edge_type)

    if entry_idx is None:
        entry_idx = min(graph.nodes)
    if exit_idx is None:
        exits = [n for n, deg in out_deg_map.items() if deg == 0]
        exit_idx = exits[0] if exits else entry_idx
    return graph, entry_idx, exit_idx


def _cfg_fn_rows(
    ctx: CfgFnContext,
) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    is_dag = nx.is_directed_acyclic_graph(ctx.graph)
    loops = _loop_stats(ctx.sccs)
    longest_path_len = _cfg_longest_path(ctx.graph, ctx.entry_idx, is_dag=is_dag)
    avg_spl = _cfg_avg_shortest_path(ctx.graph, ctx.entry_idx)
    branching = _branching_stats(ctx.graph)
    bc, closeness, eig, dom_depth, dom_frontier_sizes = _compute_cfg_centralities(
        ctx.graph, ctx.entry_idx
    )
    loop_nodes = _loop_nodes(ctx.sccs)
    out_deg_map = _degree_dict(ctx.graph, direction="out")
    in_deg_map = _degree_dict(ctx.graph, direction="in")

    fn_row = (
        _to_decimal(ctx.fn_goid),
        ctx.repo,
        ctx.commit,
        ctx.rel_path,
        ctx.module,
        ctx.qualname,
        ctx.graph.number_of_nodes(),
        ctx.graph.number_of_edges(),
        not is_dag,
        len(ctx.sccs),
        longest_path_len,
        avg_spl,
        branching[0],
        branching[1],
        branching[2],
        max(dom_depth.values()) if dom_depth else None,
        (
            (sum(dom_frontier_sizes.values()) / len(dom_frontier_sizes))
            if dom_frontier_sizes
            else 0.0
        ),
        max(dom_frontier_sizes.values()) if dom_frontier_sizes else 0,
        loops[0],
        loops[1],
        max(bc.values()) if bc else 0.0,
        (sum(bc.values()) / len(bc)) if bc else 0.0,
        (sum(closeness.values()) / len(closeness)) if closeness else 0.0,
        max(eig.values()) if eig else 0.0,
        ctx.now,
        1,
    )

    block_rows: list[tuple[object, ...]] = []
    for node, data in ctx.graph.nodes(data=True):
        block_rows.append(
            (
                _to_decimal(ctx.fn_goid),
                ctx.repo,
                ctx.commit,
                node,
                data.get("kind") == "entry",
                data.get("kind") == "exit",
                out_deg_map.get(node, 0) > 1,
                in_deg_map.get(node, 0) > 1,
                dom_depth.get(node),
                None,
                bc.get(node, 0.0),
                closeness.get(node, 0.0),
                eig.get(node, 0.0),
                node in loop_nodes,
                False,
                None,
                ctx.now,
                1,
            )
        )
    return fn_row, block_rows


def _cfg_rows_for_fn(
    *,
    fn_goid: int,
    meta: tuple[str, str | None, str | None],
    inputs: CfgInputs,
) -> CfgFnRows | None:
    blocks = inputs.blocks_by_fn.get(fn_goid, [])
    edges = inputs.edges_by_fn.get(fn_goid, [])
    if not blocks:
        return None
    graph, entry_idx, exit_idx = _build_cfg_graph(blocks, edges)
    sccs = list(nx.strongly_connected_components(graph))
    ctx = CfgFnContext(
        repo=inputs.repo,
        commit=inputs.commit,
        fn_goid=fn_goid,
        rel_path=meta[0],
        module=meta[1],
        qualname=meta[2],
        graph=graph,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        sccs=sccs,
        now=inputs.now,
    )
    fn_row, block_rows = _cfg_fn_rows(ctx)
    ext_row = _cfg_ext_row(ctx, edges)
    return CfgFnRows(fn_row=fn_row, ext_row=ext_row, block_rows=block_rows)


def _cfg_ext_row(
    ctx: CfgFnContext,
    edges: list[tuple[int, int, str]],
) -> tuple[object, ...]:
    reachable = set(nx.descendants(ctx.graph, ctx.entry_idx)) | {ctx.entry_idx}
    unreachable_count = max(ctx.graph.number_of_nodes() - len(reachable), 0)

    back_targets = {dst for _, dst, edge_kind in edges if edge_kind == "back"}
    edge_kinds = Counter(edge_kind for _, _, edge_kind in edges)
    simple_paths = _bounded_simple_path_count(ctx.graph, {ctx.entry_idx}, {ctx.exit_idx})

    return (
        _to_decimal(ctx.fn_goid),
        ctx.repo,
        ctx.commit,
        unreachable_count,
        len(back_targets),
        edge_kinds.get("true", 0),
        edge_kinds.get("false", 0),
        edge_kinds.get("back", 0),
        edge_kinds.get("exception", 0),
        edge_kinds.get("fallthrough", 0),
        edge_kinds.get("loop", 0),
        simple_paths,
        ctx.now,
        1,
    )


def compute_cfg_metrics(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Populate cfg_function_metrics and cfg_block_metrics tables."""
    con = gateway.con
    ensure_schema(con, "analytics.cfg_function_metrics")
    ensure_schema(con, "analytics.cfg_block_metrics")
    ensure_schema(con, "analytics.cfg_function_metrics_ext")

    blocks_by_fn, edges_by_fn = _load_cfg_blocks(gateway, repo, commit)
    metadata = _function_metadata(gateway, repo, commit)
    now = datetime.now(UTC)

    fn_rows: list[tuple[object, ...]] = []
    fn_ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []
    inputs = CfgInputs(
        repo=repo,
        commit=commit,
        blocks_by_fn=blocks_by_fn,
        edges_by_fn=edges_by_fn,
        now=now,
    )

    for fn_goid, meta in metadata.items():
        rows = _cfg_rows_for_fn(
            fn_goid=fn_goid,
            meta=meta,
            inputs=inputs,
        )
        if rows is None:
            continue
        fn_rows.append(rows.fn_row)
        fn_ext_rows.append(rows.ext_row)
        block_rows.extend(rows.block_rows)

    con.execute(
        "DELETE FROM analytics.cfg_function_metrics WHERE repo = ? AND commit = ?", [repo, commit]
    )
    con.execute(
        "DELETE FROM analytics.cfg_function_metrics_ext WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.cfg_block_metrics WHERE repo = ? AND commit = ?", [repo, commit]
    )

    if fn_rows:
        con.executemany(
            """
            INSERT INTO analytics.cfg_function_metrics (
                function_goid_h128, repo, commit, rel_path, module, qualname,
                cfg_block_count, cfg_edge_count, cfg_has_cycles, cfg_scc_count,
                cfg_longest_path_len, cfg_avg_shortest_path_len,
                cfg_branching_factor_mean, cfg_branching_factor_max,
                cfg_linear_block_fraction, cfg_dom_tree_height,
                cfg_dominance_frontier_size_mean, cfg_dominance_frontier_size_max,
                cfg_loop_count, cfg_loop_nesting_depth_max,
                cfg_bc_betweenness_max, cfg_bc_betweenness_mean,
                cfg_bc_closeness_mean, cfg_bc_eigenvector_max,
                created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fn_rows,
        )
    if block_rows:
        con.executemany(
            """
            INSERT INTO analytics.cfg_block_metrics (
                function_goid_h128, repo, commit, block_idx, is_entry, is_exit,
                is_branch, is_join, dom_depth, dominates_exit, bc_betweenness,
                bc_closeness, bc_eigenvector, in_loop_scc, loop_header,
                loop_nesting_depth, created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            block_rows,
        )
    if fn_ext_rows:
        con.executemany(
            """
            INSERT INTO analytics.cfg_function_metrics_ext (
                function_goid_h128, repo, commit,
                unreachable_block_count, loop_header_count,
                true_edge_count, false_edge_count, back_edge_count,
                exception_edge_count, fallthrough_edge_count, loop_edge_count,
                entry_exit_simple_paths, created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fn_ext_rows,
        )


def compute_dfg_metrics(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Populate dfg_function_metrics and dfg_block_metrics tables."""
    con = gateway.con
    ensure_schema(con, "analytics.dfg_function_metrics")
    ensure_schema(con, "analytics.dfg_block_metrics")
    ensure_schema(con, "analytics.dfg_function_metrics_ext")

    edges_by_fn = _load_dfg_edges(gateway, repo, commit)
    metadata = _function_metadata(gateway, repo, commit)
    now = datetime.now(UTC)

    fn_rows: list[tuple[object, ...]] = []
    fn_ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []

    for fn_goid, meta in metadata.items():
        ctx = _build_dfg_context(
            DfgInputs(
                fn_goid=fn_goid,
                meta=meta,
                edges=edges_by_fn.get(fn_goid, []),
                repo=repo,
                commit=commit,
                now=now,
            )
        )
        if ctx is None:
            continue

        fn_rows.append(_dfg_fn_row(ctx))
        fn_ext_rows.append(_dfg_ext_row(ctx))
        block_rows.extend(_dfg_block_rows(ctx))

    con.execute(
        "DELETE FROM analytics.dfg_function_metrics WHERE repo = ? AND commit = ?", [repo, commit]
    )
    con.execute(
        "DELETE FROM analytics.dfg_block_metrics WHERE repo = ? AND commit = ?", [repo, commit]
    )
    con.execute(
        "DELETE FROM analytics.dfg_function_metrics_ext WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    if fn_rows:
        con.executemany(
            """
            INSERT INTO analytics.dfg_function_metrics (
                function_goid_h128, repo, commit, rel_path, module, qualname,
                dfg_block_count, dfg_edge_count, dfg_phi_edge_count, dfg_symbol_count,
                dfg_component_count, dfg_scc_count, dfg_has_cycles,
                dfg_longest_chain_len, dfg_avg_shortest_path_len,
                dfg_avg_in_degree, dfg_avg_out_degree, dfg_max_in_degree,
                dfg_max_out_degree, dfg_branchy_block_fraction,
                dfg_bc_betweenness_max, dfg_bc_betweenness_mean, dfg_bc_eigenvector_max,
                created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fn_rows,
        )
    if block_rows:
        con.executemany(
            """
            INSERT INTO analytics.dfg_block_metrics (
                function_goid_h128, repo, commit, block_idx,
                dfg_in_degree, dfg_out_degree, dfg_phi_in_degree, dfg_phi_out_degree,
                dfg_bc_betweenness, dfg_bc_closeness, dfg_bc_eigenvector,
                dfg_in_scc, dfg_in_chain, created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            block_rows,
        )
    if fn_ext_rows:
        con.executemany(
            """
            INSERT INTO analytics.dfg_function_metrics_ext (
                function_goid_h128, repo, commit,
                data_flow_edge_count, intra_block_edge_count,
                use_kind_phi_count, use_kind_data_flow_count,
                use_kind_intra_block_count, use_kind_other_count,
                phi_edge_ratio, entry_exit_simple_paths,
                created_at, metrics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fn_ext_rows,
        )
