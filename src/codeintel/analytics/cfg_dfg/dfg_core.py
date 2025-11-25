"""DFG construction and metric helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from codeintel.analytics.graph_service import (
    GraphContext,
    bounded_simple_path_count,
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
    dfg_path_lengths,
)
from codeintel.graphs.nx_views import _normalize_decimal
from codeintel.storage.gateway import DuckDBError, StorageGateway

MAX_CFG_EIGEN_SAMPLE = 200
MAX_DFG_CENTRALITY_SAMPLE = 100
MAX_SIMPLE_PATHS = 1000
MAX_PATH_CUTOFF = 50

if TYPE_CHECKING:
    import networkx as nx


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


def load_dfg_edges(
    gateway: StorageGateway, _repo: str, _commit: str
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    """
    Load DFG edges grouped by function GOID.

    Returns
    -------
    dict[int, list[tuple[int, int, str, str, bool, str]]]
        Mapping of GOID -> edge tuples.
    """
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = defaultdict(list)
    try:
        rows: Iterable[tuple[int, int, int, str, str, bool, str]] = gateway.con.execute(
            """
            SELECT function_goid_h128::BIGINT, src_block_id, dst_block_id,
                   src_var, dst_var, via_phi, use_kind
            FROM graph.dfg_edges
            """
        ).fetchall()
    except DuckDBError:
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


def _parse_block_idx(block_id: str | int | None) -> int | None:
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
    dfg_in_deg = _degree_dict(graph, direction="in")
    dfg_out_deg = _degree_dict(graph, direction="out")
    dfg_phi_in = dict.fromkeys(graph.nodes, 0)
    dfg_phi_out = dict.fromkeys(graph.nodes, 0)
    for src, dst, data in graph.edges(data=True):
        if data.get("via_phi"):
            dfg_phi_out[src] += 1
            dfg_phi_in[dst] += 1

    components_count, sccs, has_cycles = dfg_component_stats(graph)
    path_lengths = dfg_path_lengths(graph)
    centrality_ctx = replace(
        inputs.graph_ctx,
        betweenness_sample=min(inputs.graph_ctx.betweenness_sample, MAX_DFG_CENTRALITY_SAMPLE),
        eigen_max_iter=min(inputs.graph_ctx.eigen_max_iter, MAX_CFG_EIGEN_SAMPLE),
    )
    centralities = dfg_centralities(graph, ctx=centrality_ctx)

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


def dfg_function_metadata(
    gateway: StorageGateway, repo: str, commit: str
) -> dict[int, tuple[str, str | None, str | None]]:
    """
    Load function metadata for DFG metrics.

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
