"""CFG construction and metric helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.cfg_dfg.helpers import degree_dict, parse_block_idx
from codeintel.build.analytics.compute.graphs import (
    build_cfg_graph,
    cfg_centralities,
    dfg_component_stats,
)
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from datetime import datetime

    import networkx as nx

    from codeintel.build.graphs.runtime.context import GraphContext


@dataclass(frozen=True)
class CfgFnContext:
    """Context bundle for computing CFG block metric rows."""

    repo: str
    commit: str
    fn_goid: int
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
class CfgCentralityData:
    """Centrality and dominance metrics for a CFG."""

    bc: dict[int, float]
    closeness: dict[int, float]
    eig: dict[int, float]
    dom_depth: dict[int, int]
    dom_frontier_sizes: dict[int, int]
    out_deg_map: dict[int, int]
    in_deg_map: dict[int, int]


def load_cfg_blocks(
    cfg_blocks_frame: pl.DataFrame,
    cfg_edges_frame: pl.DataFrame,
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

    for row in cfg_blocks_frame.iter_rows(named=True):
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

    for row in cfg_edges_frame.iter_rows(named=True):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_idx = parse_block_idx(row.get("src_block_id"))
        dst_idx = parse_block_idx(row.get("dst_block_id"))
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


def cfg_block_rows(ctx: CfgFnContext) -> list[tuple[object, ...]]:
    """
    Build block-level CFG metrics rows.

    Returns
    -------
    list[tuple[object, ...]]
        Block metrics rows for analytics.cfg_block_metrics.
    """
    centrality_data = _compute_centrality_data(ctx.graph, ctx.entry_idx, ctx.graph_ctx)
    loop_nodes_set = loop_nodes(ctx.sccs)
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
    return block_rows


def cfg_rows_for_fn(
    *,
    fn_goid: int,
    inputs: CfgInputs,
) -> list[tuple[object, ...]] | None:
    """
    Build CFG block rows for a single function.

    Returns
    -------
    list[tuple[object, ...]] | None
        Block rows when blocks are available; otherwise None.
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
        graph=graph,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        sccs=sccs,
        now=inputs.now,
        graph_ctx=inputs.graph_ctx,
    )
    return cfg_block_rows(ctx)
