"""Native Hamilton implementation for coverage_functions target.

This module implements the coverage_functions analytics target as a pure Hamilton DAG,
computing per-function coverage metrics by joining GOIDs with coverage line data.

The compute node reads from core.goids and analytics.coverage_lines, joins them
based on file path and line ranges, then aggregates coverage metrics per function.

Includes Hamilton-native validation via @check_output_custom (Phase 6).
"""

from __future__ import annotations

import logging

import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, tag

from codeintel.analytics.compute.coverage.compute import build_coverage_functions_expr
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@tag(domain="analytics", target="coverage_functions", node_type="compute")
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "function_goid_h128",
            "urn",
            "repo",
            "commit",
            "rel_path",
            "language",
            "kind",
            "qualname",
            "start_line",
            "end_line",
            "executable_lines",
            "covered_lines",
            "coverage_ratio",
            "tested",
        ],
        no_nulls=["function_goid_h128", "repo", "commit"],
    ),
)
def t__coverage_functions__compute(env: BuildEnv) -> ir.Table | None:
    """Compute per-function coverage metrics from GOIDs and coverage lines.

    Build an Ibis expression that joins function GOIDs with coverage line data
    to aggregate executable lines, covered lines, and coverage ratios per function.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.

    Returns
    -------
    ir.Table | None
        Ibis expression for analytics.coverage_functions, or None if source
        tables cannot be accessed.

    Notes
    -----
    The expression joins core.goids (filtered to functions/methods) with
    analytics.coverage_lines based on:
    - Same repo, commit, and rel_path
    - Coverage line number between function start_line and end_line

    Output columns (16 total):
    - function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname
    - start_line, end_line, executable_lines, covered_lines, coverage_ratio
    - tested, untested_reason, created_at
    """
    return build_coverage_functions_expr(env.gateway, env.snapshot)


@tag(domain="analytics", target="coverage_functions", node_type="materialize")
def t__coverage_functions(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_functions__compute: ir.Table | None,
) -> TargetRunRecord:
    """Materialize coverage_functions compute result to DuckDB.

    Write the Ibis expression from the compute node to analytics.coverage_functions,
    creating a DatasetRef for lineage tracking.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__coverage_functions__compute
        Ibis expression for coverage_functions from compute node, or None.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "coverage_functions")

    if executor.should_skip():
        return executor.skip()

    if t__coverage_functions__compute is None:
        LOG.warning("coverage_functions: skipping - no expression to materialize")
        return executor.skip()

    def compute() -> dict[str, int]:
        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="coverage_functions",
            input_hash=executor.input_hash,
        )
        ref = materialize_table(
            ctx,
            "analytics.coverage_functions",
            t__coverage_functions__compute,
        )
        return {ref.table_key: ref.row_count or 0}

    return executor.execute(compute)


__all__ = ["t__coverage_functions", "t__coverage_functions__compute"]
