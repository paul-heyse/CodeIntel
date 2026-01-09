"""CFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.cfg_dfg.helpers import (
    cfg_blocks_rowset,
    cfg_edges_rowset,
    degree_dict,
    parse_block_idx,
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
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id

MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    from datetime import datetime

    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.core.columnar.execution_context import ExecutionContext


@dataclass(frozen=True)
class CfgFnContext:
    """Context bundle for computing CFG function and block metric rows."""

    repo: str
    commit: str
    fn_goid: int
    rel_path: str
    module: str | None
    qualname: str | None
    graph: GraphInput
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


def _list_values(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _sort_cfg_groups(
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]],
    edges_by_fn: dict[int, list[tuple[int, int, str]]],
) -> None:
    for blocks in blocks_by_fn.values():
        blocks.sort(key=lambda item: item[0])
    for edges in edges_by_fn.values():
        edges.sort(key=lambda item: (item[0], item[1], item[2]))


def load_cfg_blocks(
    cfg_blocks_frame: pa.Table,
    cfg_edges_frame: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    ctx: ExecutionContext | None = None,
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

    blocks_table = cfg_blocks_rowset(cfg_blocks_frame, repo=repo, commit=commit, ctx=ctx)
    for row in iter_rows(blocks_table):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        block_idx_values = _list_values(row.get("block_idx"))
        kind_values = _list_values(row.get("kind"))
        in_deg_values = _list_values(row.get("in_degree"))
        out_deg_values = _list_values(row.get("out_degree"))
        for block_idx_raw, kind_raw, in_deg_raw, out_deg_raw in zip(
            block_idx_values,
            kind_values,
            in_deg_values,
            out_deg_values,
            strict=False,
        ):
            block_idx = normalize_decimal_id(block_idx_raw)
            if block_idx is None:
                continue
            blocks_by_fn[int(fn_id)].append(
                (
                    int(block_idx),
                    str(kind_raw) if kind_raw is not None else "unknown",
                    int(normalize_decimal_id(in_deg_raw) or 0),
                    int(normalize_decimal_id(out_deg_raw) or 0),
                )
            )

    edges_table = cfg_edges_rowset(cfg_edges_frame, repo=repo, commit=commit, ctx=ctx)
    for row in iter_rows(edges_table):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_values = _list_values(row.get("src_block_id"))
        dst_values = _list_values(row.get("dst_block_id"))
        kind_values = _list_values(row.get("edge_kind"))
        for src_raw, dst_raw, kind_raw in zip(
            src_values,
            dst_values,
            kind_values,
            strict=False,
        ):
            src_idx = parse_block_idx(_coerce_block_id(src_raw))
            dst_idx = parse_block_idx(_coerce_block_id(dst_raw))
            if src_idx is None or dst_idx is None:
                continue
            edges_by_fn[int(fn_id)].append(
                (
                    src_idx,
                    dst_idx,
                    str(kind_raw) if kind_raw is not None else "unknown",
                )
            )
    _sort_cfg_groups(blocks_by_fn, edges_by_fn)

    return blocks_by_fn, edges_by_fn


def branching_stats(graph: GraphInput) -> tuple[float, int, float]:
    """
    Return branching mean, max, and linear fraction for a CFG.

    Returns
    -------
    tuple[float, int, float]
        Mean branching factor, maximum branching factor, and linear block fraction.
    """
    store = ensure_store(graph)
    in_degrees = degree_dict(graph, direction="in")
    out_degrees_map = degree_dict(graph, direction="out")
    out_degrees = [deg for deg in out_degrees_map.values() if deg > 0]
    branching_mean = (sum(out_degrees) / len(out_degrees)) if out_degrees else 0.0
    branching_max = max(out_degrees) if out_degrees else 0
    linear_blocks: list[int] = []
    for node in store.node_ids():
        node_idx = int(str(node))
        if in_degrees.get(node_idx, 0) == 1 and out_degrees_map.get(node_idx, 0) == 1:
            linear_blocks.append(node_idx)
    linear_fraction = (
        len(linear_blocks) / store.graph.num_nodes() if store.graph.num_nodes() else 0.0
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
    graph: GraphInput,
    entry_idx: int,
    graph_ctx: GraphContext,
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
    centrality_data = _compute_centrality_data(ctx.graph, ctx.entry_idx, ctx.graph_ctx)
    graph_store = ensure_store(ctx.graph)
    fn_row = _cfg_function_row(ctx, graph_store, centrality_data)
    block_rows = _cfg_block_rows(ctx, graph_store, centrality_data)
    return fn_row, block_rows


def _cfg_function_row(
    ctx: CfgFnContext,
    store: RxGraphStore,
    centrality_data: CfgCentralityData,
) -> tuple[object, ...]:
    """Build the function-level CFG metrics row.

    Returns
    -------
    tuple[object, ...]
        Function metrics row matching analytics.cfg_function_metrics schema.
    """
    sccs = ctx.sccs
    loops = loop_stats(sccs)
    has_cycles = any(len(comp) > 1 for comp in sccs)
    is_dag = not has_cycles
    longest_path_len = cfg_longest_path_length(ctx.graph, ctx.entry_idx, is_dag=is_dag)
    avg_spl = cfg_avg_shortest_path_length(ctx.graph, ctx.entry_idx)
    branching = branching_stats(ctx.graph)
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
    return (
        ctx.fn_goid,
        ctx.repo,
        ctx.commit,
        ctx.rel_path,
        ctx.module,
        ctx.qualname,
        store.graph.num_nodes(),
        store.graph.num_edges(),
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


def _cfg_block_rows(
    ctx: CfgFnContext,
    store: RxGraphStore,
    centrality_data: CfgCentralityData,
) -> list[tuple[object, ...]]:
    """Build block-level CFG metrics rows.

    Returns
    -------
    list[tuple[object, ...]]
        Block metrics rows matching analytics.cfg_block_metrics schema.
    """
    loop_nodes_set = loop_nodes(ctx.sccs)
    rows: list[tuple[object, ...]] = []
    for node in store.node_ids():
        node_idx = int(str(node))
        data = store.get_node_attrs(node)
        rows.append(
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
    return rows


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
    unreachable_count = max(graph_node_count(ctx.graph) - len(reachable), 0)

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
