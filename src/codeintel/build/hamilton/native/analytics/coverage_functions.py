"""Native Hamilton implementation for coverage_functions target.

This module implements the coverage_functions analytics target as a pure Hamilton DAG,
computing per-function coverage metrics by joining GOIDs with coverage line data.

The compute node reads from core.goids and analytics.coverage_lines, joins them
based on file path and line ranges, then aggregates coverage metrics per function.

Includes Hamilton-native validation via @check_output_custom (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.compute.coverage.compute import build_coverage_functions_expr
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node("analytics.coverage_functions"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("coverage_functions"),
    table_key=value("analytics.coverage_functions"),
)
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
@tag(
    domain="analytics",
    target="coverage_functions",
    node_type="compute",
    target_="t__coverage_functions__compute",
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
    m__analytics__coverage_functions: dict[str, Any],
) -> TargetRunRecord:
    """Finalize coverage_functions execution from DAG-visible DuckDB materialization.

    The DuckDB write is performed by a Hamilton materializer node
    (``m__analytics__coverage_functions``). This target node converts the saver
    metadata into a TargetRunRecord and persists the manifest on success.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    m__analytics__coverage_functions
        Materialization metadata dict produced by the DuckDB saver node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="coverage_functions",
        expected_table_key="analytics.coverage_functions",
        materialization=m__analytics__coverage_functions,
    )


__all__ = ["t__coverage_functions", "t__coverage_functions__compute"]
