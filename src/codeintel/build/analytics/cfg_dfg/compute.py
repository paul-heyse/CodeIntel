"""Pure compute functions for CFG/DFG metrics.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/cfg_dfg.py`.

The functions extract control-flow and data-flow graph metrics per function,
returning structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.cfg_dfg.cfg_core import (
    CfgFnRows,
    CfgInputs,
    cfg_rows_for_fn,
    load_cfg_blocks,
)
from codeintel.build.analytics.cfg_dfg.dfg_core import (
    DfgInputs,
    build_dfg_context,
    dfg_block_rows,
    dfg_ext_row,
    dfg_fn_row,
    load_dfg_edges,
)
from codeintel.build.analytics.cfg_dfg.helpers import load_function_metadata
from codeintel.build.analytics.graphs.constants import (
    MAX_CFG_CENTRALITY_SAMPLE,
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.tabular.conversion import tabular_to_frame

if TYPE_CHECKING:
    from codeintel.build.tabular.types import InferableTabularInput
    from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True)
class CfgMetricsResult:
    """Result container for CFG metrics computation.

    Contains row data for all three CFG tables without performing writes.
    The rows are tuples matching the column specifications in the schema.

    Attributes
    ----------
    fn_rows
        Rows for analytics.cfg_function_metrics table.
    block_rows
        Rows for analytics.cfg_block_metrics table.
    ext_rows
        Rows for analytics.cfg_function_metrics_ext table.
    """

    fn_rows: tuple[tuple[object, ...], ...]
    block_rows: tuple[tuple[object, ...], ...]
    ext_rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class DfgMetricsResult:
    """Result container for DFG metrics computation.

    Contains row data for all three DFG tables without performing writes.
    The rows are tuples matching the column specifications in the schema.

    Attributes
    ----------
    fn_rows
        Rows for analytics.dfg_function_metrics table.
    block_rows
        Rows for analytics.dfg_block_metrics table.
    ext_rows
        Rows for analytics.dfg_function_metrics_ext table.
    """

    fn_rows: tuple[tuple[object, ...], ...]
    block_rows: tuple[tuple[object, ...], ...]
    ext_rows: tuple[tuple[object, ...], ...]


def compute_cfg_metrics_pure(
    snapshot: SnapshotRef,
    cfg_blocks_input: InferableTabularInput,
    cfg_edges_input: InferableTabularInput,
    goids_input: InferableTabularInput,
    modules_input: InferableTabularInput,
) -> CfgMetricsResult:
    """Compute CFG metrics without writing to database.

    Extract control-flow graph metrics for all functions in the snapshot,
    returning structured row data that can be materialized separately.

    Parameters
    ----------
    cfg_blocks_input
        Tabular input for ``graph.cfg_blocks``.
    cfg_edges_input
        Tabular input for ``graph.cfg_edges``.
    goids_input
        Tabular input for ``core.goids``.
    modules_input
        Tabular input for ``core.modules``.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    CfgMetricsResult
        Container with rows for cfg_function_metrics, cfg_block_metrics,
        and cfg_function_metrics_ext tables.

    Notes
    -----
    This function is a pure transformation that reads from the database but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.
    """
    cfg_blocks = tabular_to_frame(cfg_blocks_input)
    cfg_edges = tabular_to_frame(cfg_edges_input)
    goids = tabular_to_frame(goids_input)
    modules = tabular_to_frame(modules_input)
    blocks_by_fn, edges_by_fn = load_cfg_blocks(
        cfg_blocks,
        cfg_edges,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    metadata = load_function_metadata(goids, modules, repo=snapshot.repo, commit=snapshot.commit)
    metrics_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=snapshot.repo,
            commit=snapshot.commit,
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
        repo=snapshot.repo,
        commit=snapshot.commit,
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

    return CfgMetricsResult(
        fn_rows=tuple(fn_rows),
        block_rows=tuple(block_rows),
        ext_rows=tuple(fn_ext_rows),
    )


def compute_dfg_metrics_pure(
    dfg_edges_input: InferableTabularInput,
    goids_input: InferableTabularInput,
    modules_input: InferableTabularInput,
    repo: str,
    commit: str,
) -> DfgMetricsResult:
    """Compute DFG metrics without writing to database.

    Extract data-flow graph metrics for all functions in the snapshot,
    returning structured row data that can be materialized separately.

    Parameters
    ----------
    dfg_edges_input
        Tabular input for ``graph.dfg_edges``.
    goids_input
        Tabular input for ``core.goids``.
    modules_input
        Tabular input for ``core.modules``.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    DfgMetricsResult
        Container with rows for dfg_function_metrics, dfg_block_metrics,
        and dfg_function_metrics_ext tables.

    Notes
    -----
    This function is a pure transformation that reads from the database but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.
    """
    dfg_edges = tabular_to_frame(dfg_edges_input)
    goids = tabular_to_frame(goids_input)
    modules = tabular_to_frame(modules_input)
    edges_by_fn = load_dfg_edges(
        dfg_edges,
        repo=repo,
        commit=commit,
    )
    metadata = load_function_metadata(goids, modules, repo=repo, commit=commit)
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

    return DfgMetricsResult(
        fn_rows=tuple(fn_rows),
        block_rows=tuple(block_rows),
        ext_rows=tuple(fn_ext_rows),
    )


__all__ = [
    "CfgMetricsResult",
    "DfgMetricsResult",
    "compute_cfg_metrics_pure",
    "compute_dfg_metrics_pure",
]
