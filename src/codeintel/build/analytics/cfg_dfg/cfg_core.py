"""CFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.cfg_dfg.helpers import (
    degree_dict,
    parse_block_idx,
    prefilter_table,
)
from codeintel.build.analytics.compute.graphs import (
    bounded_simple_path_count,
    build_cfg_graph,
    cfg_avg_shortest_path_length,
    cfg_centralities,
    cfg_longest_path_length,
    cfg_reachable_nodes,
    dfg_component_stats,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id

MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    from datetime import datetime

    import networkx as nx

    from codeintel.build.graphs.runtime.context import GraphContext


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


def _coerce_block_id(value: object) -> str | int | None:
    if isinstance(value, (str, int)):
        return value
    return None


def load_cfg_blocks(
    cfg_blocks_frame: pa.Table,
    cfg_edges_frame: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
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

    filtered_blocks = prefilter_table(
        cfg_blocks_frame,
        repo=repo,
        commit=commit,
        require_valid=("function_goid_h128", "block_idx"),
    )
    for row in iter_rows(filtered_blocks):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        block_idx = normalize_decimal_id(row.get("block_idx"))
        if fn_id is None or block_idx is None:
            continue
        kind = row.get("kind")
        in_deg = normalize_decimal_id(row.get("in_degree")) or 0
        out_deg = normalize_decimal_id(row.get("out_degree")) or 0
        blocks_by_fn[int(fn_id)].append(
            (
                int(block_idx),
                str(kind) if kind is not None else "unknown",
                int(in_deg),
                int(out_deg),
            )
        )

    filtered_edges = prefilter_table(
        cfg_edges_frame,
        repo=repo,
        commit=commit,
        require_valid=("function_goid_h128", "src_block_id", "dst_block_id"),
    )
    for row in iter_rows(filtered_edges):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_idx = parse_block_idx(_coerce_block_id(row.get("src_block_id")))
        dst_idx = parse_block_idx(_coerce_block_id(row.get("dst_block_id")))
        if src_idx is None or dst_idx is None:
            continue
        edge_kind = row.get("edge_kind")
        edges_by_fn[int(fn_id)].append(
            (
                src_idx,
                dst_idx,
                str(edge_kind) if edge_kind is not None else "unknown",
            )
        )

    return blocks_by_fn, edges_by_fn


def branching_stats(graph: nx.DiGraph) -> tuple[float, int, float]:
    """
    Return branching mean, max, and linear fraction for a CFG.

    Returns
    -------
    tuple[float, int, float]
        Mean branching factor, maximum branching factor, and linear block fraction.
    """
    in_degrees = degree_dict(graph, direction="in")
    out_degrees_map = degree_dict(graph, direction="out")
    out_degrees = [deg for deg in out_degrees_map.values() if deg > 0]
    branching_mean = (sum(out_degrees) / len(out_degrees)) if out_degrees else 0.0
    branching_max = max(out_degrees) if out_degrees else 0
    linear_blocks: list[int] = []
    for node in graph.nodes:
        node_idx = int(str(node))
        if in_degrees.get(node_idx, 0) == 1 and out_degrees_map.get(node_idx, 0) == 1:
            linear_blocks.append(node_idx)
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
        out_deg_map=degree_dict(graph, direction="out"),
        in_deg_map=degree_dict(graph, direction="in"),
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
        node_idx = int(str(node))
        block_rows.append(
            (
                ctx.fn_goid,
                ctx.repo,
                ctx.commit,
                node_idx,
                data.get("kind") == "entry",
                data.get("kind") == "exit",
                centrality_data.out_deg_map.get(node_idx, 0) > 1,
                centrality_data.in_deg_map.get(node_idx, 0) > 1,
                centrality_data.dom_depth.get(node_idx),
                None,
                centrality_data.bc.get(node_idx, 0.0),
                centrality_data.closeness.get(node_idx, 0.0),
                centrality_data.eig.get(node_idx, 0.0),
                node_idx in loop_nodes_set,
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
