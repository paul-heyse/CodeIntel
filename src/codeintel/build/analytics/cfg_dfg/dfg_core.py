"""DFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.cfg_dfg.helpers import (
    degree_dict,
    dfg_edges_rowset,
    parse_block_idx,
)
from codeintel.build.analytics.compute.graphs import (
    bounded_simple_path_count,
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
    dfg_path_lengths,
    normalize_dfg_graph,
)
from codeintel.build.analytics.graphs.constants import (
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.build.graphs.rx.algos import in_degree_by_id, out_degree_by_id
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id

MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    from datetime import datetime

    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.core.columnar.execution_context import ExecutionContext


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
    graph: RxGraphStore
    edges: list[tuple[int, int, str, str, bool, str]]
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
    ctx: ExecutionContext | None = None,
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    """
    Load DFG edges grouped by function GOID.

    Returns
    -------
    dict[int, list[tuple[int, int, str, str, bool, str]]]
        Mapping of GOID -> edge tuples.
    """
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    edges_table = dfg_edges_rowset(dfg_edges_frame, repo=repo, commit=commit, ctx=ctx)
    for row in iter_rows(edges_table):
        fn_id = normalize_decimal_id(row.get("function_goid_h128"))
        if fn_id is None:
            continue
        src_values = _list_values(row.get("src_block_id"))
        dst_values = _list_values(row.get("dst_block_id"))
        src_vars = _list_values(row.get("src_var"))
        dst_vars = _list_values(row.get("dst_var"))
        via_phi_values = _list_values(row.get("via_phi"))
        use_kinds = _list_values(row.get("use_kind"))
        for src_raw, dst_raw, src_var, dst_var, via_phi, use_kind in zip(
            src_values,
            dst_values,
            src_vars,
            dst_vars,
            via_phi_values,
            use_kinds,
            strict=False,
        ):
            src_idx = parse_block_idx(_coerce_block_id(src_raw))
            dst_idx = parse_block_idx(_coerce_block_id(dst_raw))
            if src_idx is None or dst_idx is None:
                continue
            if not isinstance(src_var, str) or not isinstance(dst_var, str):
                continue
            edges_by_fn[int(fn_id)].append(
                (
                    src_idx,
                    dst_idx,
                    src_var,
                    dst_var,
                    bool(via_phi),
                    str(use_kind) if use_kind is not None else "unknown",
                )
            )
    return edges_by_fn


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
    analysis_graph = (
        normalize_dfg_graph(graph, inputs.edges)
        if inputs.graph_ctx.enable_dfg_normalization
        else graph
    )
    dfg_in_deg = degree_dict(analysis_graph, direction="in")
    dfg_out_deg = degree_dict(analysis_graph, direction="out")
    dfg_phi_in = {int(str(node_id)): 0 for node_id in graph.node_ids()}
    dfg_phi_out = {int(str(node_id)): 0 for node_id in graph.node_ids()}
    for src, dst, _src_var, _dst_var, via_phi, _use_kind in inputs.edges:
        if not via_phi:
            continue
        dfg_phi_out[src] = dfg_phi_out.get(src, 0) + 1
        dfg_phi_in[dst] = dfg_phi_in.get(dst, 0) + 1

    component_stats = dfg_component_stats(analysis_graph)
    path_lengths = dfg_path_lengths(analysis_graph)
    centralities = dfg_centralities(
        analysis_graph,
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
        edges=list(inputs.edges),
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
        ctx.graph.graph.num_nodes(),
        ctx.graph.graph.num_edges(),
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
    for node_id in ctx.graph.node_ids():
        node_idx = int(str(node_id))
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
    edge_kinds = Counter(edge[5] for edge in ctx.edges)
    data_flow_edges = edge_kinds.get("data-flow", 0)
    intra_block_edges = edge_kinds.get("intra-block", 0)
    phi_edges = sum(1 for edge in ctx.edges if edge[4])
    total_edges = ctx.graph.graph.num_edges() or 1
    phi_ratio = phi_edges / total_edges
    other_kinds = sum(
        count
        for kind, count in edge_kinds.items()
        if kind not in {"data-flow", "intra-block", "phi"}
    )

    in_degrees = in_degree_by_id(ctx.graph)
    out_degrees = out_degree_by_id(ctx.graph)
    sources = {node_id for node_id in ctx.graph.node_ids() if in_degrees.get(node_id, 0) == 0}
    sinks = {node_id for node_id in ctx.graph.node_ids() if out_degrees.get(node_id, 0) == 0}
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
