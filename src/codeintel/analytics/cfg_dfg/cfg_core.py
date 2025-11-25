"""CFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

from codeintel.analytics.graph_service import (
    GraphContext,
    bounded_simple_path_count,
    build_cfg_graph,
    cfg_avg_shortest_path_length,
    cfg_centralities,
    cfg_longest_path_length,
    cfg_reachable_nodes,
    dfg_component_stats,
)
from codeintel.graphs.nx_views import _normalize_decimal
from codeintel.storage.gateway import DuckDBError, StorageGateway

MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    import networkx as nx


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
    graph_ctx: GraphContext


@dataclass(frozen=True)
class CfgInputs:
    """Shared CFG inputs for per-function metric derivation."""

    repo: str
    commit: str
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]]
    edges_by_fn: dict[int, list[tuple[int, int, str]]]
    now: datetime
    graph_ctx: GraphContext


@dataclass(frozen=True)
class CfgFnRows:
    """Container for per-function CFG rows."""

    fn_row: tuple[object, ...]
    ext_row: tuple[object, ...]
    block_rows: list[tuple[object, ...]]


@dataclass(frozen=True)
class CfgCentralityData:
    """Centrality and dominance metrics for a CFG."""

    bc: dict[int, float]
    closeness: dict[int, float]
    eig: dict[int, float]
    dom_depth: dict[int, int]
    dom_frontier_sizes: dict[int, int]
    out_deg_map: dict[int, int]
    in_deg_map: dict[int, int]


def _degree_dict(
    graph: nx.DiGraph,
    *,
    direction: str,
    weight: str | None = None,
) -> dict[int, int]:
    """
    Materialize degree counts into a concrete mapping for type safety.

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


def load_cfg_blocks(
    gateway: StorageGateway, _repo: str, _commit: str
) -> tuple[dict[int, list[tuple[int, str, int, int]]], dict[int, list[tuple[int, int, str]]]]:
    """
    Load CFG blocks and edges grouped by function GOID.

    Returns
    -------
    tuple[dict[int, list[tuple[int, str, int, int]]], dict[int, list[tuple[int, int, str]]]]
        Blocks and edges grouped by GOID.
    """
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    edges_by_fn: dict[int, list[tuple[int, int, str]]] = defaultdict(list)

    try:
        block_rows: Iterable[tuple[int, int, str, int, int]] = gateway.con.execute(
            """
            SELECT function_goid_h128::BIGINT, block_idx, kind, in_degree, out_degree
            FROM graph.cfg_blocks
            """
        ).fetchall()
    except DuckDBError:
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
    except DuckDBError:
        return blocks_by_fn, edges_by_fn
    for fn, src_id, dst_id, edge_type in edge_rows:
        src_idx = parse_block_idx(src_id) if src_id is not None else None
        dst_idx = parse_block_idx(dst_id) if dst_id is not None else None
        if src_idx is None or dst_idx is None:
            continue
        edges_by_fn[int(fn)].append((src_idx, dst_idx, str(edge_type)))

    return blocks_by_fn, edges_by_fn


def parse_block_idx(block_id: str | int | None) -> int | None:
    """
    Extract the integer block index from a block identifier.

    Returns
    -------
    int | None
        Parsed block index when available.
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


def branching_stats(graph: nx.DiGraph) -> tuple[float, int, float]:
    """
    Return branching mean, max, and linear fraction for a CFG.

    Returns
    -------
    tuple[float, int, float]
        Mean branching factor, maximum branching factor, and linear block fraction.
    """
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


def loop_stats(sccs: list[set[int]]) -> tuple[int, int]:
    """
    Return loop count and maximum loop size.

    Returns
    -------
    tuple[int, int]
        Loop count and maximum strongly connected component size.
    """
    loop_sccs = [comp for comp in sccs if len(comp) > 1]
    loop_count = len(loop_sccs)
    loop_max = max((len(comp) for comp in loop_sccs), default=0)
    return loop_count, loop_max


def loop_nodes(sccs: list[set[int]]) -> set[int]:
    """
    Return nodes participating in loops.

    Returns
    -------
    set[int]
        Nodes belonging to loop SCCs.
    """
    return set().union(*[comp for comp in sccs if len(comp) > 1]) if sccs else set()


def _compute_centrality_data(
    graph: nx.DiGraph, entry_idx: int, graph_ctx: GraphContext
) -> CfgCentralityData:
    """
    Compute centrality and dominance data for a CFG.

    Returns
    -------
    CfgCentralityData
        Aggregated centrality metrics for the graph.
    """
    centrality, dominance = cfg_centralities(
        graph,
        entry_idx,
        ctx=graph_ctx,
    )
    return CfgCentralityData(
        bc=centrality.betweenness,
        closeness=centrality.closeness,
        eig=centrality.eigenvector,
        dom_depth=dominance.depth,
        dom_frontier_sizes=dominance.frontier_sizes,
        out_deg_map=_degree_dict(graph, direction="out"),
        in_deg_map=_degree_dict(graph, direction="in"),
    )


def cfg_fn_rows(
    ctx: CfgFnContext,
) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    """
    Build function-level and block-level CFG rows.

    Returns
    -------
    tuple[tuple[object, ...], list[tuple[object, ...]]]
        Function metrics row and block metrics rows.
    """
    sccs = ctx.sccs
    loops = loop_stats(sccs)
    has_cycles = any(len(comp) > 1 for comp in sccs)
    is_dag = not has_cycles
    longest_path_len = cfg_longest_path_length(ctx.graph, ctx.entry_idx, is_dag=is_dag)
    avg_spl = cfg_avg_shortest_path_length(ctx.graph, ctx.entry_idx)
    branching = branching_stats(ctx.graph)
    centrality_data = _compute_centrality_data(ctx.graph, ctx.entry_idx, ctx.graph_ctx)
    loop_nodes_set = loop_nodes(sccs)
    dom_frontier_mean = (
        sum(centrality_data.dom_frontier_sizes.values()) / len(centrality_data.dom_frontier_sizes)
        if centrality_data.dom_frontier_sizes
        else 0.0
    )
    dom_frontier_max = (
        max(centrality_data.dom_frontier_sizes.values())
        if centrality_data.dom_frontier_sizes
        else 0
    )

    fn_row = (
        ctx.fn_goid,
        ctx.repo,
        ctx.commit,
        ctx.rel_path,
        ctx.module,
        ctx.qualname,
        ctx.graph.number_of_nodes(),
        ctx.graph.number_of_edges(),
        has_cycles,
        len(sccs),
        longest_path_len,
        avg_spl,
        branching[0],
        branching[1],
        branching[2],
        max(centrality_data.dom_depth.values()) if centrality_data.dom_depth else None,
        dom_frontier_mean,
        dom_frontier_max,
        loops[0],
        loops[1],
        max(centrality_data.bc.values()) if centrality_data.bc else 0.0,
        (sum(centrality_data.bc.values()) / len(centrality_data.bc)) if centrality_data.bc else 0.0,
        (sum(centrality_data.closeness.values()) / len(centrality_data.closeness))
        if centrality_data.closeness
        else 0.0,
        max(centrality_data.eig.values()) if centrality_data.eig else 0.0,
        ctx.now,
        1,
    )

    block_rows: list[tuple[object, ...]] = []
    for node, data in ctx.graph.nodes(data=True):
        block_rows.append(
            (
                ctx.fn_goid,
                ctx.repo,
                ctx.commit,
                node,
                data.get("kind") == "entry",
                data.get("kind") == "exit",
                centrality_data.out_deg_map.get(node, 0) > 1,
                centrality_data.in_deg_map.get(node, 0) > 1,
                centrality_data.dom_depth.get(node),
                None,
                centrality_data.bc.get(node, 0.0),
                centrality_data.closeness.get(node, 0.0),
                centrality_data.eig.get(node, 0.0),
                node in loop_nodes_set,
                False,
                None,
                ctx.now,
                1,
            )
        )
    return fn_row, block_rows


def cfg_ext_row(
    ctx: CfgFnContext,
    edges: list[tuple[int, int, str]],
) -> tuple[object, ...]:
    """
    Build CFG extension metrics row capturing reachability and edge kinds.

    Returns
    -------
    tuple[object, ...]
        Row matching analytics.cfg_function_metrics_ext schema.
    """
    reachable = cfg_reachable_nodes(ctx.graph, ctx.entry_idx)
    unreachable_count = max(ctx.graph.number_of_nodes() - len(reachable), 0)

    back_targets = {dst for _, dst, edge_kind in edges if edge_kind == "back"}
    edge_kinds = Counter(edge_kind for _, _, edge_kind in edges)
    simple_paths = bounded_simple_path_count(
        ctx.graph,
        {ctx.entry_idx},
        {ctx.exit_idx},
        max_paths=MAX_SIMPLE_PATHS,
        cutoff=MAX_PATH_CUTOFF,
    )

    return (
        ctx.fn_goid,
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


def cfg_rows_for_fn(
    *,
    fn_goid: int,
    meta: tuple[str, str | None, str | None],
    inputs: CfgInputs,
) -> CfgFnRows | None:
    """
    Build CFG rows for a single function.

    Returns
    -------
    CfgFnRows | None
        Structured rows when blocks are available; otherwise None.
    """
    blocks = inputs.blocks_by_fn.get(fn_goid, [])
    edges = inputs.edges_by_fn.get(fn_goid, [])
    if not blocks:
        return None
    graph, entry_idx, exit_idx = build_cfg_graph(blocks, edges)
    _, sccs, _ = dfg_component_stats(graph)
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
        graph_ctx=inputs.graph_ctx,
    )
    fn_row, block_rows = cfg_fn_rows(ctx)
    ext_row = cfg_ext_row(ctx, edges)
    return CfgFnRows(fn_row=fn_row, ext_row=ext_row, block_rows=block_rows)


def function_metadata(
    gateway: StorageGateway, repo: str, commit: str
) -> dict[int, tuple[str, str | None, str | None]]:
    """
    Load function metadata keyed by GOID.

    Returns
    -------
    dict[int, tuple[str, str | None, str | None]]
        Mapping of GOID -> (rel_path, module, qualname).
    """
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
