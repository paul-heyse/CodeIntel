"""Materialize CFG/DFG analytics tables."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from codeintel.analytics.cfg_dfg.cfg_core import (
    CfgFnRows,
    CfgInputs,
    cfg_rows_for_fn,
    load_cfg_blocks,
)
from codeintel.analytics.cfg_dfg.cfg_core import (
    function_metadata as cfg_function_metadata,
)
from codeintel.analytics.cfg_dfg.dfg_core import (
    DfgInputs,
    build_dfg_context,
    dfg_block_rows,
    dfg_ext_row,
    dfg_fn_row,
    dfg_function_metadata,
    load_dfg_edges,
)
from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import GraphContext
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

MAX_CFG_CENTRALITY_SAMPLE = 100
MAX_CFG_EIGEN_SAMPLE = 200
MAX_DFG_CENTRALITY_SAMPLE = 100


def compute_cfg_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
) -> None:
    """Populate cfg_function_metrics and cfg_block_metrics tables."""
    if context is not None and (context.repo != repo or context.commit != commit):
        return
    con = gateway.con
    ensure_schema(con, "analytics.cfg_function_metrics")
    ensure_schema(con, "analytics.cfg_block_metrics")
    ensure_schema(con, "analytics.cfg_function_metrics_ext")

    blocks_by_fn, edges_by_fn = load_cfg_blocks(gateway, repo, commit)
    metadata = cfg_function_metadata(gateway, repo, commit)
    ctx_base = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
    )
    resolved_now = ctx_base.resolved_now()
    metrics_ctx = replace(
        ctx_base,
        now=resolved_now,
        betweenness_sample=min(ctx_base.betweenness_sample, MAX_CFG_CENTRALITY_SAMPLE),
        eigen_max_iter=min(ctx_base.eigen_max_iter, MAX_CFG_EIGEN_SAMPLE),
    )

    fn_rows: list[tuple[object, ...]] = []
    fn_ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []
    inputs = CfgInputs(
        repo=repo,
        commit=commit,
        blocks_by_fn=blocks_by_fn,
        edges_by_fn=edges_by_fn,
        now=resolved_now,
        graph_ctx=metrics_ctx,
    )

    for fn_goid, meta in metadata.items():
        rows: CfgFnRows | None = cfg_rows_for_fn(
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
        "DELETE FROM analytics.cfg_function_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.cfg_function_metrics_ext WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.cfg_block_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
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


def compute_dfg_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
) -> None:
    """Populate dfg_function_metrics and dfg_block_metrics tables."""
    if context is not None and (context.repo != repo or context.commit != commit):
        return
    con = gateway.con
    ensure_schema(con, "analytics.dfg_function_metrics")
    ensure_schema(con, "analytics.dfg_block_metrics")
    ensure_schema(con, "analytics.dfg_function_metrics_ext")

    edges_by_fn = load_dfg_edges(gateway, repo, commit)
    metadata = dfg_function_metadata(gateway, repo, commit)
    ctx_base = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
    )
    resolved_now = ctx_base.resolved_now()
    metrics_ctx = replace(
        ctx_base,
        now=resolved_now,
        betweenness_sample=min(ctx_base.betweenness_sample, MAX_DFG_CENTRALITY_SAMPLE),
        eigen_max_iter=min(ctx_base.eigen_max_iter, MAX_CFG_EIGEN_SAMPLE),
    )

    fn_rows: list[tuple[object, ...]] = []
    fn_ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []

    for fn_goid, meta in metadata.items():
        ctx = build_dfg_context(
            DfgInputs(
                fn_goid=fn_goid,
                meta=meta,
                edges=edges_by_fn.get(fn_goid, []),
                repo=repo,
                commit=commit,
                now=resolved_now,
                graph_ctx=metrics_ctx,
            )
        )
        if ctx is None:
            continue

        fn_rows.append(dfg_fn_row(ctx))
        fn_ext_rows.append(dfg_ext_row(ctx))
        block_rows.extend(dfg_block_rows(ctx))

    con.execute(
        "DELETE FROM analytics.dfg_function_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.dfg_block_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
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
