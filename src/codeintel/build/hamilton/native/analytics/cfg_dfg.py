"""Native Hamilton implementation for cfg_dfg_metrics target.

This module provides the Hamilton native nodes for CFG/DFG metrics:
- `t__cfg_dfg_metrics__compute_cfg`: Pure compute node for CFG metrics
- `t__cfg_dfg_metrics__compute_dfg`: Pure compute node for DFG metrics
- `t__cfg_dfg_metrics`: Materialize node that writes all 6 tables

The compute nodes call pure functions from `codeintel.analytics.cfg_dfg.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.
"""

from __future__ import annotations

from hamilton.function_modifiers import tag

from codeintel.analytics.cfg_dfg.compute import (
    CfgMetricsResult,
    DfgMetricsResult,
    compute_cfg_metrics_pure,
    compute_dfg_metrics_pure,
)
from codeintel.analytics.cfg_dfg.materialize import (
    CFG_BLOCK_METRICS_COLS,
    CFG_FUNCTION_METRICS_COLS,
    CFG_FUNCTION_METRICS_EXT_COLS,
    DFG_BLOCK_METRICS_COLS,
    DFG_FUNCTION_METRICS_COLS,
    DFG_FUNCTION_METRICS_EXT_COLS,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, CfgMetricsResult, DfgMetricsResult)


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute")
def t__cfg_dfg_metrics__compute_cfg(env: BuildEnv) -> CfgMetricsResult:
    """Compute CFG metrics for all functions in the snapshot.

    This is a pure compute node with no side effects. It reads CFG block
    data and function metadata from the database and computes control-flow
    graph metrics for each function.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    CfgMetricsResult
        Container with rows for cfg_function_metrics, cfg_block_metrics,
        and cfg_function_metrics_ext tables.

    Notes
    -----
    The metrics computed include:
    - Block and edge counts
    - Cycle detection and SCC analysis
    - Path length statistics
    - Branching factor analysis
    - Dominance tree metrics
    - Loop analysis
    - Centrality measures (betweenness, closeness, eigenvector)
    """
    return compute_cfg_metrics_pure(
        env.gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute")
def t__cfg_dfg_metrics__compute_dfg(env: BuildEnv) -> DfgMetricsResult:
    """Compute DFG metrics for all functions in the snapshot.

    This is a pure compute node with no side effects. It reads DFG edge
    data and function metadata from the database and computes data-flow
    graph metrics for each function.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    DfgMetricsResult
        Container with rows for dfg_function_metrics, dfg_block_metrics,
        and dfg_function_metrics_ext tables.

    Notes
    -----
    The metrics computed include:
    - Block and edge counts
    - PHI edge analysis
    - Symbol and component counts
    - SCC and cycle analysis
    - Path length statistics
    - Degree analysis (in/out)
    - Centrality measures (betweenness, closeness, eigenvector)
    """
    return compute_dfg_metrics_pure(
        env.gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="materialize")
def t__cfg_dfg_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult,
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult,
) -> TargetRunRecord:
    """Materialize all 6 CFG/DFG tables to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed CFG and DFG metrics to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__cfg_dfg_metrics__compute_cfg
        Computed CFG metrics from the cfg compute node.
    t__cfg_dfg_metrics__compute_dfg
        Computed DFG metrics from the dfg compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.cfg_function_metrics
    - analytics.cfg_block_metrics
    - analytics.cfg_function_metrics_ext
    - analytics.dfg_function_metrics
    - analytics.dfg_block_metrics
    - analytics.dfg_function_metrics_ext
    """
    executor = NativeTargetExecutor.for_target(env, graph, "cfg_dfg_metrics")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure tables exist
        backend = env.gateway.policy
        backend.ensure_table("analytics.cfg_function_metrics")
        backend.ensure_table("analytics.cfg_block_metrics")
        backend.ensure_table("analytics.cfg_function_metrics_ext")
        backend.ensure_table("analytics.dfg_function_metrics")
        backend.ensure_table("analytics.dfg_block_metrics")
        backend.ensure_table("analytics.dfg_function_metrics_ext")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="cfg_dfg_metrics",
            input_hash=executor.input_hash,
        )

        row_counts: dict[str, int] = {}

        # Materialize CFG tables
        cfg = t__cfg_dfg_metrics__compute_cfg
        cfg_tables = [
            ("analytics.cfg_function_metrics", cfg.fn_rows, CFG_FUNCTION_METRICS_COLS),
            ("analytics.cfg_block_metrics", cfg.block_rows, CFG_BLOCK_METRICS_COLS),
            ("analytics.cfg_function_metrics_ext", cfg.ext_rows, CFG_FUNCTION_METRICS_EXT_COLS),
        ]
        for table_key, rows, cols in cfg_tables:
            ref = materialize_rows(ctx, table_key, rows, cols)
            row_counts[table_key] = ref.row_count or 0

        # Materialize DFG tables
        dfg = t__cfg_dfg_metrics__compute_dfg
        dfg_tables = [
            ("analytics.dfg_function_metrics", dfg.fn_rows, DFG_FUNCTION_METRICS_COLS),
            ("analytics.dfg_block_metrics", dfg.block_rows, DFG_BLOCK_METRICS_COLS),
            ("analytics.dfg_function_metrics_ext", dfg.ext_rows, DFG_FUNCTION_METRICS_EXT_COLS),
        ]
        for table_key, rows, cols in dfg_tables:
            ref = materialize_rows(ctx, table_key, rows, cols)
            row_counts[table_key] = ref.row_count or 0

        return row_counts

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__cfg_dfg_metrics",
    "t__cfg_dfg_metrics__compute_cfg",
    "t__cfg_dfg_metrics__compute_dfg",
]
