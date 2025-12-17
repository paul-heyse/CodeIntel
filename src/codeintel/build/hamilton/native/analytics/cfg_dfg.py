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

from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

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
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, CfgMetricsResult, DfgMetricsResult)


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute")
def t__cfg_dfg_metrics__compute_cfg(env: BuildEnv, graph: TargetGraph) -> CfgMetricsResult | None:
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
    CfgMetricsResult | None
        Container with rows for cfg_function_metrics, cfg_block_metrics,
        and cfg_function_metrics_ext tables.
        Returns None when manifest-skip indicates the target is current.

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
    target = graph.get("cfg_dfg_metrics")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return compute_cfg_metrics_pure(
        env.gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute")
def t__cfg_dfg_metrics__compute_dfg(env: BuildEnv, graph: TargetGraph) -> DfgMetricsResult | None:
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
    DfgMetricsResult | None
        Container with rows for dfg_function_metrics, dfg_block_metrics,
        and dfg_function_metrics_ext tables.
        Returns None when manifest-skip indicates the target is current.

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
    target = graph.get("cfg_dfg_metrics")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return compute_dfg_metrics_pure(
        env.gateway,
        env.snapshot.repo,
        env.snapshot.commit,
    )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.cfg_function_metrics"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.cfg_function_metrics"),
    columns=value(tuple(CFG_FUNCTION_METRICS_COLS)),
)
@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute", target_="cfg_dfg_metrics__cfg_function_metrics_rows")
def cfg_dfg_metrics__cfg_function_metrics_rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_function_metrics."""
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.fn_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.cfg_block_metrics"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.cfg_block_metrics"),
    columns=value(tuple(CFG_BLOCK_METRICS_COLS)),
)
@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute", target_="cfg_dfg_metrics__cfg_block_metrics_rows")
def cfg_dfg_metrics__cfg_block_metrics_rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_block_metrics."""
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.block_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.cfg_function_metrics_ext"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.cfg_function_metrics_ext"),
    columns=value(tuple(CFG_FUNCTION_METRICS_EXT_COLS)),
)
@tag(
    domain="analytics",
    target="cfg_dfg_metrics",
    node_type="compute",
    target_="cfg_dfg_metrics__cfg_function_metrics_ext_rows",
)
def cfg_dfg_metrics__cfg_function_metrics_ext_rows(
    t__cfg_dfg_metrics__compute_cfg: CfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.cfg_function_metrics_ext."""
    if t__cfg_dfg_metrics__compute_cfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_cfg.ext_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.dfg_function_metrics"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.dfg_function_metrics"),
    columns=value(tuple(DFG_FUNCTION_METRICS_COLS)),
)
@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute", target_="cfg_dfg_metrics__dfg_function_metrics_rows")
def cfg_dfg_metrics__dfg_function_metrics_rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_function_metrics."""
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.fn_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.dfg_block_metrics"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.dfg_block_metrics"),
    columns=value(tuple(DFG_BLOCK_METRICS_COLS)),
)
@tag(domain="analytics", target="cfg_dfg_metrics", node_type="compute", target_="cfg_dfg_metrics__dfg_block_metrics_rows")
def cfg_dfg_metrics__dfg_block_metrics_rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_block_metrics."""
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.block_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.dfg_function_metrics_ext"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("cfg_dfg_metrics"),
    table_key=value("analytics.dfg_function_metrics_ext"),
    columns=value(tuple(DFG_FUNCTION_METRICS_EXT_COLS)),
)
@tag(
    domain="analytics",
    target="cfg_dfg_metrics",
    node_type="compute",
    target_="cfg_dfg_metrics__dfg_function_metrics_ext_rows",
)
def cfg_dfg_metrics__dfg_function_metrics_ext_rows(
    t__cfg_dfg_metrics__compute_dfg: DfgMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.dfg_function_metrics_ext."""
    if t__cfg_dfg_metrics__compute_dfg is None:
        return None
    return tuple(t__cfg_dfg_metrics__compute_dfg.ext_rows)


@tag(domain="analytics", target="cfg_dfg_metrics", node_type="materialize")
def t__cfg_dfg_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__cfg_function_metrics: dict[str, Any],
    m__analytics__cfg_block_metrics: dict[str, Any],
    m__analytics__cfg_function_metrics_ext: dict[str, Any],
    m__analytics__dfg_function_metrics: dict[str, Any],
    m__analytics__dfg_block_metrics: dict[str, Any],
    m__analytics__dfg_function_metrics_ext: dict[str, Any],
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
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="cfg_dfg_metrics",
        materializations={
            "analytics.cfg_function_metrics": m__analytics__cfg_function_metrics,
            "analytics.cfg_block_metrics": m__analytics__cfg_block_metrics,
            "analytics.cfg_function_metrics_ext": m__analytics__cfg_function_metrics_ext,
            "analytics.dfg_function_metrics": m__analytics__dfg_function_metrics,
            "analytics.dfg_block_metrics": m__analytics__dfg_block_metrics,
            "analytics.dfg_function_metrics_ext": m__analytics__dfg_function_metrics_ext,
        },
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__cfg_dfg_metrics",
    "t__cfg_dfg_metrics__compute_cfg",
    "t__cfg_dfg_metrics__compute_dfg",
]
