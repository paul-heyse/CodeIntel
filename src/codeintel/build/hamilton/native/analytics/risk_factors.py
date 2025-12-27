"""Native Hamilton implementation for risk_factors target.

This module demonstrates Phase 3 native execution with:
- Pure Ibis compute node (no side effects)
- Explicit materialize node (side-effect boundary)
- TargetRunRecord creation with proper dataset refs
- Hamilton-native validation via @check_output_custom (Phase 1.5 POC)
- Schema documentation via @schema.output
"""

from __future__ import annotations

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    schema,
    source,
    step,
)

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    IbisTableSaveSpec,
    SaverContext,
    save_ibis_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord, options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hamilton.validators import (
    build_enum_column_contract,
    build_table_contract,
)
from codeintel.core.ibis_typing import add, cast_dtype, ge, gt, ibis_bool, truediv

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, ir.Table)

RISK_FACTORS_TARGET_NAME = "risk_factors"
RISK_FACTORS_TABLE_KEY = "analytics.goid_risk_factors"
RISK_FACTORS_TABLE_KEYS = (RISK_FACTORS_TABLE_KEY,)
RISK_FACTORS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=RISK_FACTORS_TARGET_NAME,
)


COMPLEXITY_THRESHOLD = 10
FAN_IN_BUCKET_SIZE = 5
FAN_OUT_BUCKET_SIZE = 10
NO_TESTS_PENALTY = 3
RISK_LEVEL_HIGH_THRESHOLD = 5
RISK_LEVEL_MEDIUM_THRESHOLD = 3


@tag_compute(domain="analytics", target=RISK_FACTORS_TARGET_NAME)
def risk_factors__fan_in(q__graph__call_graph_edges: ir.Table) -> ir.Table:
    """Compute fan-in counts per callee function.

    Parameters
    ----------
    q__graph__call_graph_edges
        Ibis table expression for graph.call_graph_edges.

    Returns
    -------
    ir.Table
        Ibis expression with columns:
        - function_goid_h128
        - fan_in_count
    """
    return (
        q__graph__call_graph_edges.group_by("callee_goid_h128")
        .aggregate(fan_in_count=ibis._.count())
        .rename({"function_goid_h128": "callee_goid_h128"})
    )


@tag_compute(domain="analytics", target=RISK_FACTORS_TARGET_NAME)
def risk_factors__fan_out(q__graph__call_graph_edges: ir.Table) -> ir.Table:
    """Compute fan-out counts per caller function.

    Parameters
    ----------
    q__graph__call_graph_edges
        Ibis table expression for graph.call_graph_edges.

    Returns
    -------
    ir.Table
        Ibis expression with columns:
        - function_goid_h128
        - fan_out_count
    """
    return (
        q__graph__call_graph_edges.group_by("caller_goid_h128")
        .aggregate(fan_out_count=ibis._.count())
        .rename({"function_goid_h128": "caller_goid_h128"})
    )


def _risk_factors_join_metrics(
    metrics: ir.Table,
    coverage: ir.Table,
    fan_in: ir.Table,
    fan_out: ir.Table,
) -> ir.Table:
    """Join function metrics with coverage and fan-in/fan-out counts.

    Parameters
    ----------
    metrics
        Ibis table expression for analytics.function_metrics.
    coverage
        Ibis table expression for analytics.coverage_functions.
    fan_in
        Ibis table expression with per-function fan-in counts.
    fan_out
        Ibis table expression with per-function fan-out counts.

    Returns
    -------
    ir.Table
        Joined Ibis expression with metrics and centrality columns.
    """
    joined = metrics.left_join(
        coverage,
        predicates=[
            ibis_bool(metrics.function_goid_h128 == coverage.function_goid_h128),
            ibis_bool(metrics.repo == coverage.repo),
            ibis_bool(metrics.commit == coverage.commit),
        ],
    )
    has_tests = ibis.coalesce(coverage.tested, False)
    risk = joined.select(
        "function_goid_h128",
        "repo",
        "commit",
        "cyclomatic_complexity",
        has_tests=has_tests,
    )

    risk = risk.left_join(
        fan_in,
        predicates=[ibis_bool(risk.function_goid_h128 == fan_in.function_goid_h128)],
    ).select(
        risk.function_goid_h128,
        risk.repo,
        risk.commit,
        risk.cyclomatic_complexity,
        risk.has_tests,
        fan_in_count=ibis.coalesce(fan_in.fan_in_count, 0),
    )

    return risk.left_join(
        fan_out,
        predicates=[ibis_bool(risk.function_goid_h128 == fan_out.function_goid_h128)],
    ).select(
        risk.function_goid_h128,
        risk.repo,
        risk.commit,
        risk.cyclomatic_complexity,
        risk.has_tests,
        risk.fan_in_count,
        fan_out_count=ibis.coalesce(fan_out.fan_out_count, 0),
    )


def _risk_factors_score(risk: ir.Table) -> ir.Table:
    """Compute the risk_score column from complexity and centrality signals.

    Parameters
    ----------
    risk
        Joined metrics table with fan-in and fan-out counts.

    Returns
    -------
    ir.Table
        Ibis expression with an additional ``risk_score`` column.
    """
    risk_score = ibis.cases(
        (gt(risk.cyclomatic_complexity, COMPLEXITY_THRESHOLD), 2),
        else_=0,
    )
    risk_score = add(
        risk_score, cast_dtype(truediv(risk.fan_in_count, FAN_IN_BUCKET_SIZE), "int64")
    )
    risk_score = add(
        risk_score, cast_dtype(truediv(risk.fan_out_count, FAN_OUT_BUCKET_SIZE), "int64")
    )
    risk_score = ibis.cases(
        (risk.has_tests, risk_score),
        else_=add(risk_score, NO_TESTS_PENALTY),
    )

    return risk.mutate(risk_score=risk_score)


def _risk_factors_finalize(risk: ir.Table) -> ir.Table:
    """Select final columns and categorize the risk_level.

    Parameters
    ----------
    risk
        Metrics table with ``risk_score`` computed.

    Returns
    -------
    ir.Table
        Final Ibis expression for analytics.goid_risk_factors.
    """
    return risk.select(
        "function_goid_h128",
        "repo",
        "commit",
        "risk_score",
        risk_level=ibis.cases(
            (ge(risk.risk_score, RISK_LEVEL_HIGH_THRESHOLD), "high"),
            (ge(risk.risk_score, RISK_LEVEL_MEDIUM_THRESHOLD), "medium"),
            else_="low",
        ),
        cyclomatic_complexity=risk.cyclomatic_complexity,
        fan_in_count=risk.fan_in_count,
        fan_out_count=risk.fan_out_count,
        has_tests=risk.has_tests,
    )




@save_ibis_table(
    context=RISK_FACTORS_SAVE_CONTEXT,
    spec=IbisTableSaveSpec(table_key=RISK_FACTORS_TABLE_KEY),
)
@pipe_input(
    step(
        _risk_factors_join_metrics,
        fan_in=source("risk_factors__fan_in"),
        fan_out=source("risk_factors__fan_out"),
        coverage=source("q__analytics__coverage_functions"),
    ),
    step(_risk_factors_score),
    step(_risk_factors_finalize),
    namespace=None,
    on_input="q__analytics__function_metrics",
)
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "function_goid_h128",
            "repo",
            "commit",
            "risk_score",
            "risk_level",
            "cyclomatic_complexity",
            "fan_in_count",
            "fan_out_count",
            "has_tests",
        ],
        no_nulls=["function_goid_h128", "repo", "commit"],
    ),
    *build_enum_column_contract(
        column="risk_level",
        allowed_values={"high", "medium", "low"},
    ),
)
@tag_compute(
    domain="analytics",
    target=RISK_FACTORS_TARGET_NAME,
    target_="t__risk_factors__compute",
)
@schema.output(
    ("function_goid_h128", "string"),
    ("repo", "string"),
    ("commit", "string"),
    ("risk_score", "int"),
    ("risk_level", "string"),
    ("cyclomatic_complexity", "int"),
    ("fan_in_count", "int"),
    ("fan_out_count", "int"),
    ("has_tests", "bool"),
    target_="t__risk_factors__compute",
)
def t__risk_factors__compute(
    q__analytics__function_metrics: ir.Table,
) -> ir.Table:
    """Compute risk factors from function metrics and call graph.

    This is a pure Ibis transformation with no side effects. It combines
    function complexity metrics with call graph centrality to identify
    high-risk functions.

    Parameters
    ----------
    q__analytics__function_metrics
        Ibis table expression for function metrics.

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
    # NOTE: The @pipe_input chain transforms q__analytics__function_metrics into
    # the final result; returning it keeps the function body minimal and ensures
    # the intermediate steps are DAG-visible.
    return q__analytics__function_metrics


risk_factors__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=RISK_FACTORS_TARGET_NAME,
    table_keys=RISK_FACTORS_TABLE_KEYS,
)


@codeintel_target(domain="analytics", target=RISK_FACTORS_TARGET_NAME)
def t__risk_factors(
    env: BuildEnv,
    catalog: DagCatalog,
    risk_factors__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Compute composite risk factors per function.

    The actual DuckDB write is performed by a Hamilton materializer node
    (``m__analytics__goid_risk_factors``). This target node converts the
    materialization results into a TargetRunRecord and persists the manifest
    on success.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    risk_factors__table_materializations
        Materialization results for analytics.goid_risk_factors.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=RISK_FACTORS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=risk_factors__table_materializations,
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__risk_factors",
    "t__risk_factors__compute",
]
