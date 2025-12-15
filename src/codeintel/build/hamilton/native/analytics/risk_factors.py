"""Native Hamilton implementation for risk_factors target.

This module demonstrates Phase 3 native execution with:
- Pure Ibis compute node (no side effects)
- Explicit materialize node (side-effect boundary)
- TargetRunRecord creation with proper dataset refs
"""

from __future__ import annotations

from typing import Any, cast

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import ge, gt

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


COMPLEXITY_THRESHOLD = 10
FAN_IN_BUCKET_SIZE = 5
FAN_OUT_BUCKET_SIZE = 10
NO_TESTS_PENALTY = 3
RISK_LEVEL_HIGH_THRESHOLD = 5
RISK_LEVEL_MEDIUM_THRESHOLD = 3


@tag(domain="analytics", target="risk_factors", node_type="compute")
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute risk factors from function metrics and call graph.

    This is a pure Ibis transformation with no side effects. It combines
    function complexity metrics with call graph centrality to identify
    high-risk functions.

    Parameters
    ----------
    q__analytics__function_metrics
        Ibis table expression for function metrics.
    q__graph__call_graph_edges
        Ibis table expression for call graph edges.

    Returns
    -------
    ir.Table
        Ibis expression for analytics.goid_risk_factors table.

    Notes
    -----
    Risk score is computed as:
    - High complexity (cyclomatic > 10): +2 points
    - High fan-in (called by many): +1 point per 5 callers
    - High fan-out (calls many): +1 point per 10 callees
    - No tests: +3 points
    """
    # Compute fan-in (how many functions call this one)
    fan_in = (
        q__graph__call_graph_edges.group_by("callee_goid_h128")
        .aggregate(fan_in_count=ibis._.count())
        .rename({"callee_goid_h128": "function_goid_h128"})
    )

    # Compute fan-out (how many functions this one calls)
    fan_out = (
        q__graph__call_graph_edges.group_by("caller_goid_h128")
        .aggregate(fan_out_count=ibis._.count())
        .rename({"caller_goid_h128": "function_goid_h128"})
    )

    # Start with function metrics
    risk = q__analytics__function_metrics.select(
        "function_goid_h128",
        "repo",
        "commit",
        "cyclomatic_complexity",
        "has_tests",
    )

    # Join fan-in and fan-out
    risk = risk.left_join(
        fan_in, cast("Any", risk.function_goid_h128 == fan_in.function_goid_h128)
    ).select(
        risk.function_goid_h128,
        risk.repo,
        risk.commit,
        risk.cyclomatic_complexity,
        risk.has_tests,
        fan_in_count=ibis.coalesce(fan_in.fan_in_count, 0),
    )

    risk = risk.left_join(
        fan_out, cast("Any", risk.function_goid_h128 == fan_out.function_goid_h128)
    ).select(
        risk.function_goid_h128,
        risk.repo,
        risk.commit,
        risk.cyclomatic_complexity,
        risk.has_tests,
        risk.fan_in_count,
        fan_out_count=ibis.coalesce(fan_out.fan_out_count, 0),
    )

    # Calculate risk score
    risk_score = ibis.cases(
        (gt(risk.cyclomatic_complexity, COMPLEXITY_THRESHOLD), 2),
        else_=0,
    )
    risk_score += cast("Any", cast("Any", risk.fan_in_count) / FAN_IN_BUCKET_SIZE).cast("int64")
    risk_score += cast("Any", cast("Any", risk.fan_out_count) / FAN_OUT_BUCKET_SIZE).cast("int64")
    risk_score = ibis.cases(
        (risk.has_tests, risk_score),
        else_=risk_score + NO_TESTS_PENALTY,
    )

    # Final selection with risk_level categorization
    return risk.select(
        "function_goid_h128",
        "repo",
        "commit",
        risk_score=risk_score,
        risk_level=ibis.cases(
            (ge(risk_score, RISK_LEVEL_HIGH_THRESHOLD), "high"),
            (ge(risk_score, RISK_LEVEL_MEDIUM_THRESHOLD), "medium"),
            else_="low",
        ),
        cyclomatic_complexity=risk.cyclomatic_complexity,
        fan_in_count=risk.fan_in_count,
        fan_out_count=risk.fan_out_count,
        has_tests=risk.has_tests,
    )


@tag(domain="analytics", target="risk_factors", node_type="materialize")
def t__risk_factors(
    env: BuildEnv,
    graph: TargetGraph,
    t__risk_factors__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize risk_factors compute result to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed Ibis expression to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__risk_factors__compute
        Computed Ibis expression from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        ref = materialize_table(
            MaterializationContext(
                gateway=env.gateway,
                snapshot=env.snapshot,
                validate=env.validate_outputs,
            ),
            "analytics.goid_risk_factors",
            t__risk_factors__compute,
        )
        return {ref.table_key: ref.row_count or 0}

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__risk_factors",
    "t__risk_factors__compute",
]
