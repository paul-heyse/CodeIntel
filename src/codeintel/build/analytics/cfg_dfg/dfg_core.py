"""DFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.cfg_dfg.helpers import (
    degree_dict,
    parse_block_idx,
    prefilter_table,
)
from codeintel.build.analytics.compute.graphs import (
    bounded_simple_path_count,
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
    dfg_path_lengths,
)
from codeintel.build.analytics.graphs.constants import (
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id

MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    from datetime import datetime

    import networkx as nx

    from codeintel.build.graphs.runtime.context import GraphContext


def _to_decimal(value: int) -> Decimal:
    return Decimal(value)


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
    graph_ctx: GraphContext


def load_dfg_edges(
    dfg_edges_frame: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    """
    Load DFG edges grouped by function GOID.

    Returns
    -------
    dict[int, list[tuple[int, int, str, str, bool, str]]]
        Mapping of GOID -> edge tuples.
    """
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    filtered_edges = prefilter_table(
        dfg_edges_frame,
        repo=repo,
        commit=commit,
        require_valid=("function_goid_h128", "src_block_id", "dst_block_id", "src_var", "dst_var"),
    )
    for row in iter_rows(filtered_edges):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_idx = parse_block_idx(_coerce_block_id(row.get("src_block_id")))
        dst_idx = parse_block_idx(_coerce_block_id(row.get("dst_block_id")))
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


def _coerce_block_id(value: object) -> str | int | None:
    if isinstance(value, (str, int)):
        return value
    return None


def build_dfg_context(inputs: DfgInputs) -> DfgFnContext | None:
    """
    Assemble a DFG context for a single function.

    Returns
    -------
    DfgFnContext | None
        DFG context when edges exist; otherwise None.
    """
    meta = inputs.meta
    if not inputs.edges:
        return None

    graph, phi_edges, symbol_count = build_dfg_graph(inputs.edges)
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

    component_stats = dfg_component_stats(graph)
    path_lengths = dfg_path_lengths(graph)
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
        rel_path=meta[0],
        module=meta[1],
        qualname=meta[2],
        graph=graph,
        phi_edges=phi_edges,
        symbol_count=symbol_count,
        components_count=component_stats[0],
        sccs=component_stats[1],
        has_cycles=component_stats[2],
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


def dfg_fn_row(ctx: DfgFnContext) -> tuple[object, ...]:
    """
    Build the function-level DFG metrics row.

    Returns
    -------
    tuple[object, ...]
        Row matching analytics.dfg_function_metrics schema.
    """
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


def dfg_ext_row(ctx: DfgFnContext) -> tuple[object, ...]:
    """
    Build the DFG extension metrics row.

    Returns
    -------
    tuple[object, ...]
        Row matching analytics.dfg_function_metrics_ext schema.
    """
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
    simple_paths = bounded_simple_path_count(
        ctx.graph,
        sources,
        sinks,
        max_paths=MAX_SIMPLE_PATHS,
        cutoff=MAX_PATH_CUTOFF,
    )

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
