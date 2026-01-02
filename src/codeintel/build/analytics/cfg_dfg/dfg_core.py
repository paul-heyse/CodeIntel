"""DFG construction and metric helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.cfg_dfg.helpers import degree_dict, parse_block_idx
from codeintel.build.analytics.compute.graphs import (
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
)
from codeintel.build.analytics.graphs.constants import (
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from datetime import datetime

    import networkx as nx

    from codeintel.build.graphs.runtime.context import GraphContext


def _to_decimal(value: int) -> Decimal:
    return Decimal(value)


@dataclass(frozen=True)
class DfgFnContext:
    """Context bundle for computing DFG block metric rows."""

    repo: str
    commit: str
    fn_goid: int
    graph: nx.DiGraph
    sccs: list[set[int]]
    dfg_in_deg: dict[int, int]
    dfg_out_deg: dict[int, int]
    dfg_phi_in: dict[int, int]
    dfg_phi_out: dict[int, int]
    bc: dict[int, float]
    eig: dict[int, float]
    now: datetime


@dataclass(frozen=True)
class DfgInputs:
    """Input payload for building a DFG metrics context."""

    fn_goid: int
    edges: list[tuple[int, int, str, str, bool, str]]
    repo: str
    commit: str
    now: datetime
    graph_ctx: GraphContext


def load_dfg_edges(
    dfg_edges_frame: pl.DataFrame,
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    """
    Load DFG edges grouped by function GOID.

    Returns
    -------
    dict[int, list[tuple[int, int, str, str, bool, str]]]
        Mapping of GOID -> edge tuples.
    """
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    for row in dfg_edges_frame.iter_rows(named=True):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_idx = parse_block_idx(row.get("src_block_id"))
        dst_idx = parse_block_idx(row.get("dst_block_id"))
        if src_idx is None or dst_idx is None:
            continue
        src_var = row.get("src_var")
        dst_var = row.get("dst_var")
        if not isinstance(src_var, str) or not isinstance(dst_var, str):
            continue
        use_kind = row.get("use_kind")
        edges_by_fn[int(fn_id)].append(
            (
                src_idx,
                dst_idx,
                src_var,
                dst_var,
                bool(row.get("via_phi")),
                str(use_kind) if use_kind is not None else "unknown",
            )
        )
    return edges_by_fn


def build_dfg_context(inputs: DfgInputs) -> DfgFnContext | None:
    """
    Assemble a DFG context for a single function.

    Returns
    -------
    DfgFnContext | None
        DFG context when edges exist; otherwise None.
    """
    if not inputs.edges:
        return None

    graph, _, _ = build_dfg_graph(inputs.edges)
    dfg_in_deg = degree_dict(graph, direction="in")
    dfg_out_deg = degree_dict(graph, direction="out")
    dfg_phi_in = {int(str(node)): 0 for node in graph.nodes}
    dfg_phi_out = {int(str(node)): 0 for node in graph.nodes}
    for src, dst, data in graph.edges(data=True):
        src_idx = int(str(src))
        dst_idx = int(str(dst))
        if data.get("via_phi"):
            dfg_phi_out[src_idx] += 1
            dfg_phi_in[dst_idx] += 1

    _, sccs, _ = dfg_component_stats(graph)
    centralities = dfg_centralities(
        graph,
        ctx=replace(
            inputs.graph_ctx,
            betweenness_sample=min(
                inputs.graph_ctx.betweenness_sample,
                MAX_DFG_CENTRALITY_SAMPLE,
            ),
            eigen_max_iter=min(
                inputs.graph_ctx.eigen_max_iter,
                MAX_CFG_EIGEN_SAMPLE,
            ),
        ),
    )

    return DfgFnContext(
        repo=inputs.repo,
        commit=inputs.commit,
        fn_goid=inputs.fn_goid,
        graph=graph,
        sccs=sccs,
        dfg_in_deg=dfg_in_deg,
        dfg_out_deg=dfg_out_deg,
        dfg_phi_in=dfg_phi_in,
        dfg_phi_out=dfg_phi_out,
        bc=centralities[0],
        eig=centralities[1],
        now=inputs.now,
    )


def dfg_block_rows(ctx: DfgFnContext) -> list[tuple[object, ...]]:
    """
    Build block-level DFG metrics rows.

    Returns
    -------
    list[tuple[object, ...]]
        Rows matching analytics.dfg_block_metrics schema.
    """
    loop_nodes = {node for comp in ctx.sccs if len(comp) > 1 for node in comp}
    rows: list[tuple[object, ...]] = []
    for node in ctx.graph.nodes:
        node_idx = int(str(node))
        rows.append(
            (
                _to_decimal(ctx.fn_goid),
                ctx.repo,
                ctx.commit,
                node_idx,
                ctx.dfg_in_deg.get(node_idx, 0),
                ctx.dfg_out_deg.get(node_idx, 0),
                ctx.dfg_phi_in.get(node_idx, 0),
                ctx.dfg_phi_out.get(node_idx, 0),
                ctx.bc.get(node_idx, 0.0),
                None,
                ctx.eig.get(node_idx, 0.0),
                node_idx in loop_nodes,
                False,
                ctx.now,
                1,
            )
        )
    return rows
