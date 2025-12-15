"""Native Hamilton implementation for call_graph_views target.

This module implements derived views over the call graph as a pure Hamilton DAG,
computing useful aggregate metrics and patterns from the raw call graph edges.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_tables
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@tag(domain="graphs", target="call_graph_views", node_type="compute", view="function_call_counts")
def call_graph_function_call_counts(
    env: BuildEnv,
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute per-function call count statistics from call graph edges.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__graph__call_graph_edges
        Ibis table expression for graph.call_graph_edges.

    Returns
    -------
    ir.Table
        Ibis expression with schema:
        - function_goid_h128: Function GOID hash
        - num_callees: Number of functions this function calls
        - num_unique_callees: Number of unique functions called
        - num_callers: Number of functions that call this function

    Examples
    --------
    >>> # This compute node aggregates call patterns per function
    """
    LOG.info("Computing function call counts from call graph")

    # Filter to current snapshot
    edges = q__graph__call_graph_edges.filter(
        cast(
            "Any",
            and_predicates(
                q__graph__call_graph_edges.repo == env.snapshot.repo,
                q__graph__call_graph_edges.commit == env.snapshot.commit,
            ),
        )
    )

    # Aggregate call stats per caller
    callee_stats: ir.Table = edges.group_by(
        function_goid_h128=edges.caller_goid_h128,
    ).aggregate(
        num_callees=ibis._.count(),
        num_unique_callees=cast("Any", edges.callee_goid_h128).nunique(),
    )

    # Aggregate caller stats per callee (exclude null callees)
    caller_stats: ir.Table = (
        edges.filter(cast("Any", edges.callee_goid_h128.notnull()))
        .group_by(
            function_goid_h128=edges.callee_goid_h128,
        )
        .aggregate(
            num_callers=ibis._.count(),
        )
    )

    # Full outer join to get union of caller/callee populations
    result = callee_stats.join(
        caller_stats,
        predicates=[callee_stats.function_goid_h128 == caller_stats.function_goid_h128],
        how="outer",
    )

    call_counts = result.select(
        repo=ibis.literal(env.snapshot.repo),
        commit=ibis.literal(env.snapshot.commit),
        function_goid_h128=cast("Any", result.function_goid_h128),
        num_callees=cast("Any", result.num_callees).fillna(ibis.literal(0)),
        num_unique_callees=cast("Any", result.num_unique_callees).fillna(ibis.literal(0)),
        num_callers=cast("Any", result.num_callers).fillna(ibis.literal(0)),
    )

    LOG.info("function_call_counts compute complete")
    return call_counts


@tag(domain="graphs", target="call_graph_views", node_type="compute", view="call_depth_stats")
def call_graph_depth_stats(
    env: BuildEnv,
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute call depth statistics (simplified version).

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__graph__call_graph_edges
        Ibis table expression for graph.call_graph_edges.

    Returns
    -------
    ir.Table
        Ibis expression with schema:
        - function_goid_h128: Function GOID hash
        - max_call_depth: Maximum observed call chain depth (simplified to 1 for direct calls)
        - is_leaf: Whether function calls no other functions

    Examples
    --------
    >>> # This compute node calculates simple depth metrics
    """
    LOG.info("Computing call depth stats from call graph")

    # Filter to current snapshot
    edges = q__graph__call_graph_edges.filter(
        cast(
            "Any",
            and_predicates(
                q__graph__call_graph_edges.repo == env.snapshot.repo,
                q__graph__call_graph_edges.commit == env.snapshot.commit,
            ),
        )
    )

    caller_funcs: ir.Table = edges.select(
        caller_function_goid_h128=edges.caller_goid_h128,
    ).distinct()
    callee_funcs: ir.Table = (
        edges.filter(cast("Any", edges.callee_goid_h128.notnull()))
        .select(
            function_goid_h128=edges.callee_goid_h128,
        )
        .distinct()
    )
    all_funcs: ir.Table = (
        caller_funcs.select(function_goid_h128=caller_funcs.caller_function_goid_h128)
        .union(callee_funcs)
        .distinct()
    )

    joined = all_funcs.left_join(
        caller_funcs,
        predicates=[all_funcs.function_goid_h128 == caller_funcs.caller_function_goid_h128],
    )
    is_leaf = cast("Any", joined.caller_function_goid_h128).isnull()
    depth_stats = joined.select(
        repo=ibis.literal(env.snapshot.repo),
        commit=ibis.literal(env.snapshot.commit),
        function_goid_h128=joined.function_goid_h128,
        max_call_depth=ibis.ifelse(is_leaf, ibis.literal(0), ibis.literal(1)),
        is_leaf=is_leaf,
    )

    LOG.info("call_depth_stats compute complete")
    return depth_stats


@tag(domain="graphs", target="call_graph_views", node_type="materialize")
def t__call_graph_views(
    env: BuildEnv,
    graph: TargetGraph,
    call_graph_function_call_counts: ir.Table,
    call_graph_depth_stats: ir.Table,
) -> TargetRunRecord:
    """Materialize all call graph views to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    call_graph_function_call_counts
        Ibis expression for function call counts view.
    call_graph_depth_stats
        Ibis expression for call depth stats view.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node materializes all view expressions to DuckDB tables
    """
    LOG.info("Materializing call_graph_views to DuckDB")

    executor = NativeTargetExecutor.for_target(env, graph, "call_graph_views")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        views_dict = {
            "graph.v_function_call_counts": call_graph_function_call_counts,
            "graph.v_call_depth_stats": call_graph_depth_stats,
        }

        dataset_refs = materialize_tables(
            MaterializationContext(
                gateway=env.gateway,
                snapshot=env.snapshot,
                validate=env.validate_outputs,
                owner_target="call_graph_views",
                input_hash=executor.input_hash,
            ),
            views_dict,
        )

        row_counts = {ref.table_key: ref.row_count or 0 for ref in dataset_refs}
        total_rows = sum(row_counts.values())
        LOG.info("call_graph_views materialization complete: %d total rows", total_rows)
        return row_counts

    return executor.execute(compute)


__all__ = [
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "t__call_graph_views",
]
