"""Materialize CFG/DFG analytics tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.cfg_dfg.cfg_core import (
    CfgInputs,
    cfg_rows_for_fn,
    load_cfg_blocks,
)
from codeintel.analytics.cfg_dfg.dfg_core import (
    DfgInputs,
    build_dfg_context,
    dfg_block_rows,
    dfg_ext_row,
    dfg_fn_row,
    load_dfg_edges,
)
from codeintel.analytics.cfg_dfg.helpers import load_function_metadata
from codeintel.analytics.graphs.constants import (
    MAX_CFG_CENTRALITY_SAMPLE,
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.analytics.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.analytics.cfg_dfg.cfg_core import (
        CfgFnRows,
    )
    from codeintel.storage.gateway import StorageGateway


CFG_FUNCTION_METRICS_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "rel_path",
    "module",
    "qualname",
    "cfg_block_count",
    "cfg_edge_count",
    "cfg_has_cycles",
    "cfg_scc_count",
    "cfg_longest_path_len",
    "cfg_avg_shortest_path_len",
    "cfg_branching_factor_mean",
    "cfg_branching_factor_max",
    "cfg_linear_block_fraction",
    "cfg_dom_tree_height",
    "cfg_dominance_frontier_size_mean",
    "cfg_dominance_frontier_size_max",
    "cfg_loop_count",
    "cfg_loop_nesting_depth_max",
    "cfg_bc_betweenness_max",
    "cfg_bc_betweenness_mean",
    "cfg_bc_closeness_mean",
    "cfg_bc_eigenvector_max",
    "created_at",
    "metrics_version",
]
CFG_BLOCK_METRICS_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "block_idx",
    "is_entry",
    "is_exit",
    "is_branch",
    "is_join",
    "dom_depth",
    "dominates_exit",
    "bc_betweenness",
    "bc_closeness",
    "bc_eigenvector",
    "in_loop_scc",
    "loop_header",
    "loop_nesting_depth",
    "created_at",
    "metrics_version",
]
CFG_FUNCTION_METRICS_EXT_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "unreachable_block_count",
    "loop_header_count",
    "true_edge_count",
    "false_edge_count",
    "back_edge_count",
    "exception_edge_count",
    "fallthrough_edge_count",
    "loop_edge_count",
    "entry_exit_simple_paths",
    "created_at",
    "metrics_version",
]


DFG_FUNCTION_METRICS_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "rel_path",
    "module",
    "qualname",
    "dfg_block_count",
    "dfg_edge_count",
    "dfg_phi_edge_count",
    "dfg_symbol_count",
    "dfg_component_count",
    "dfg_scc_count",
    "dfg_has_cycles",
    "dfg_longest_chain_len",
    "dfg_avg_shortest_path_len",
    "dfg_avg_in_degree",
    "dfg_avg_out_degree",
    "dfg_max_in_degree",
    "dfg_max_out_degree",
    "dfg_branchy_block_fraction",
    "dfg_bc_betweenness_max",
    "dfg_bc_betweenness_mean",
    "dfg_bc_eigenvector_max",
    "created_at",
    "metrics_version",
]
DFG_BLOCK_METRICS_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "block_idx",
    "dfg_in_degree",
    "dfg_out_degree",
    "dfg_phi_in_degree",
    "dfg_phi_out_degree",
    "dfg_bc_betweenness",
    "dfg_bc_closeness",
    "dfg_bc_eigenvector",
    "dfg_in_scc",
    "dfg_in_chain",
    "created_at",
    "metrics_version",
]
DFG_FUNCTION_METRICS_EXT_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "data_flow_edge_count",
    "intra_block_edge_count",
    "use_kind_phi_count",
    "use_kind_data_flow_count",
    "use_kind_intra_block_count",
    "use_kind_other_count",
    "phi_edge_ratio",
    "entry_exit_simple_paths",
    "created_at",
    "metrics_version",
]


def compute_cfg_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Populate cfg_function_metrics and cfg_block_metrics tables."""
    backend = DuckDBPolicyBackend(gateway)
    backend.ensure_table("analytics.cfg_function_metrics")
    backend.ensure_table("analytics.cfg_block_metrics")
    backend.ensure_table("analytics.cfg_function_metrics_ext")

    blocks_by_fn, edges_by_fn = load_cfg_blocks(gateway, repo, commit)
    metadata = load_function_metadata(gateway, repo, commit)
    metrics_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=False,
            now=datetime.now(UTC),
            betweenness_cap=MAX_CFG_CENTRALITY_SAMPLE,
            eigen_cap=MAX_CFG_EIGEN_SAMPLE,
        )
    )
    resolved_now = metrics_ctx.resolved_now()

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

    backend.delete_for_snapshot("analytics.cfg_function_metrics", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.cfg_function_metrics_ext", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.cfg_block_metrics", repo=repo, commit=commit)

    if fn_rows:
        gateway.ibis.write(
            "analytics.cfg_function_metrics",
            fn_rows,
            columns=CFG_FUNCTION_METRICS_COLS,
        )
    if block_rows:
        gateway.ibis.write(
            "analytics.cfg_block_metrics",
            block_rows,
            columns=CFG_BLOCK_METRICS_COLS,
        )
    if fn_ext_rows:
        gateway.ibis.write(
            "analytics.cfg_function_metrics_ext",
            fn_ext_rows,
            columns=CFG_FUNCTION_METRICS_EXT_COLS,
        )


def compute_dfg_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Populate dfg_function_metrics and dfg_block_metrics tables."""
    backend = DuckDBPolicyBackend(gateway)
    backend.ensure_table("analytics.dfg_function_metrics")
    backend.ensure_table("analytics.dfg_block_metrics")
    backend.ensure_table("analytics.dfg_function_metrics_ext")

    edges_by_fn = load_dfg_edges(gateway, repo, commit)
    metadata = load_function_metadata(gateway, repo, commit)
    metrics_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=False,
            now=datetime.now(UTC),
            betweenness_cap=MAX_DFG_CENTRALITY_SAMPLE,
            eigen_cap=MAX_CFG_EIGEN_SAMPLE,
        )
    )
    resolved_now = metrics_ctx.resolved_now()

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

    backend.delete_for_snapshot("analytics.dfg_function_metrics", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.dfg_block_metrics", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.dfg_function_metrics_ext", repo=repo, commit=commit)

    if fn_rows:
        gateway.ibis.write(
            "analytics.dfg_function_metrics",
            fn_rows,
            columns=DFG_FUNCTION_METRICS_COLS,
        )
    if block_rows:
        gateway.ibis.write(
            "analytics.dfg_block_metrics",
            block_rows,
            columns=DFG_BLOCK_METRICS_COLS,
        )
    if fn_ext_rows:
        gateway.ibis.write(
            "analytics.dfg_function_metrics_ext",
            fn_ext_rows,
            columns=DFG_FUNCTION_METRICS_EXT_COLS,
        )
