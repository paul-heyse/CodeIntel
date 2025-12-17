"""Consolidated Hamilton implementation for coverage-related analytics targets.

This module consolidates coverage analytics targets using Phase 1 templates:

- ``coverage_functions``: Per-function coverage aggregation (Ibis -> DuckDB)
- ``coverage_test_edges``: Test-to-function coverage edge computation (Pattern D)
- ``behavioral_coverage``: Heuristic behavior tag assignment for tests (Pattern D)

The coverage_functions target uses DAG-visible I/O via ``DuckDBIbisTableSaver``.
The other two targets use the ``executor_materialize`` template for simplified
materialize nodes with ``NativeTargetExecutor`` pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    source,
    step,
    tag,
    value,
)
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.compute.coverage.compute import (
    aggregate_coverage_lines,
    enrich_coverage_results,
    filter_coverage_lines_for_snapshot,
    filter_goids_for_snapshot,
    join_goids_with_coverage_lines,
)
from codeintel.analytics.testing import compute_test_coverage_edges
from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageTestEdgesResult:
    """Result from coverage test edges computation.

    Parameters
    ----------
    success
        Whether the computation succeeded.
    table_counts
        Mapping of table_key to row count.
    error
        Error message if computation failed.
    """

    success: bool
    table_counts: dict[str, int]
    error: str | None = None


@dataclass(frozen=True)
class BehavioralCoverageResult:
    """Result from behavioral coverage computation.

    Parameters
    ----------
    success
        Whether the computation succeeded.
    table_counts
        Mapping of table_key to row count.
    error
        Error message if computation failed.
    """

    success: bool
    table_counts: dict[str, int]
    error: str | None = None


# -----------------------------------------------------------------------------
# Coverage functions (Ibis -> DuckDB)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageJoinState:
    """Intermediate state for coverage_functions pipeline.

    Parameters
    ----------
    joined
        Joined GOIDs and coverage line table.
    goids
        Filtered GOIDs table used for aggregation group-by references.
    """

    joined: ir.Table
    goids: ir.Table


def _coverage_functions_filter_goids(goids: ir.Table, env: BuildEnv) -> ir.Table:
    """Filter GOIDs to the current snapshot and function-like kinds.

    Parameters
    ----------
    goids
        Ibis table expression for ``core.goids``.
    env
        Build environment providing the snapshot to filter against.

    Returns
    -------
    ir.Table
        Filtered GOIDs table.
    """
    return filter_goids_for_snapshot(goids, env.snapshot)


def _coverage_functions_join_with_coverage(
    goids: ir.Table,
    *,
    env: BuildEnv,
    coverage_lines: ir.Table,
) -> CoverageJoinState:
    """Join filtered GOIDs with coverage lines for the current snapshot.

    Parameters
    ----------
    goids
        Filtered GOIDs table.
    env
        Build environment providing the snapshot.
    coverage_lines
        Ibis table expression for ``analytics.coverage_lines``.

    Returns
    -------
    CoverageJoinState
        Joined table plus the filtered GOIDs table for downstream aggregation.
    """
    coverage_filtered = filter_coverage_lines_for_snapshot(coverage_lines, env.snapshot)
    joined = join_goids_with_coverage_lines(goids, coverage_filtered)
    return CoverageJoinState(joined=joined, goids=goids)


def _coverage_functions_aggregate(state: CoverageJoinState) -> ir.Table:
    """Aggregate coverage metrics per function.

    Parameters
    ----------
    state
        Coverage join state containing the joined table and filtered GOIDs.

    Returns
    -------
    ir.Table
        Aggregated coverage metrics table.
    """
    return aggregate_coverage_lines(state.joined, state.goids)


def _coverage_functions_enrich(aggregated: ir.Table) -> ir.Table:
    """Enrich aggregated coverage with ratios and tested flags.

    Parameters
    ----------
    aggregated
        Aggregated coverage table.

    Returns
    -------
    ir.Table
        Final coverage functions table.
    """
    return enrich_coverage_results(aggregated)


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node("analytics.coverage_functions"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("coverage_functions"),
    table_key=value("analytics.coverage_functions"),
)
@pipe_input(
    step(_coverage_functions_filter_goids, env=source("env")),
    step(
        _coverage_functions_join_with_coverage,
        env=source("env"),
        coverage_lines=source("q__analytics__coverage_lines"),
    ),
    step(_coverage_functions_aggregate),
    step(_coverage_functions_enrich),
    namespace=None,
    on_input="q__core__goids",
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
def t__coverage_functions__compute(
    q__core__goids: ir.Table,
) -> ir.Table:
    """Compute per-function coverage metrics from GOIDs and coverage lines.

    Parameters
    ----------
    q__core__goids
        Ibis table expression for core.goids.

    Returns
    -------
    ir.Table
        Ibis expression producing coverage functions rows.
    """
    return q__core__goids


@tag(domain="analytics", target="coverage_functions", node_type="materialize")
def t__coverage_functions(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__coverage_functions: dict[str, Any],
) -> TargetRunRecord:
    """Convert materialization metadata to a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__coverage_functions
        Materialization metadata for analytics.coverage_functions.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="coverage_functions",
        expected_table_key="analytics.coverage_functions",
        materialization=m__analytics__coverage_functions,
    )


# -----------------------------------------------------------------------------
# Coverage test edges (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="coverage_test_edges", node_type="tool")
def t__coverage_test_edges__compute(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> CoverageTestEdgesResult:
    """Compute test-to-function coverage edges.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    CoverageTestEdgesResult
        Result indicating success or failure with table counts.
    """
    if t__goids.status != "succeeded":
        return CoverageTestEdgesResult(
            success=False,
            table_counts={},
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    try:
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        compute_test_coverage_edges(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )

        return CoverageTestEdgesResult(
            success=True,
            table_counts={"analytics.test_coverage_edges": 0},
        )

    except Exception as exc:
        log.exception("Coverage test edges computation failed")
        return CoverageTestEdgesResult(
            success=False,
            table_counts={},
            error=str(exc),
        )


@tag(domain="analytics", target="coverage_test_edges", node_type="materialize")
def t__coverage_test_edges(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_test_edges__compute: CoverageTestEdgesResult,
) -> TargetRunRecord:
    """Materialize coverage test edges target using executor template.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__coverage_test_edges__compute
        Computed coverage edges result.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, "coverage_test_edges", t__coverage_test_edges__compute)


# -----------------------------------------------------------------------------
# Behavioral coverage (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="behavioral_coverage", node_type="tool")
def t__behavioral_coverage__compute(
    env: BuildEnv,
    t__test_profile: TargetRunRecord,
) -> BehavioralCoverageResult:
    """Assign heuristic behavior tags to tests.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__test_profile
        Upstream test_profile target result (for dependency).

    Returns
    -------
    BehavioralCoverageResult
        Result indicating success or failure with table counts.
    """
    if t__test_profile.status != "succeeded":
        return BehavioralCoverageResult(
            success=False,
            table_counts={},
            error=f"Upstream test_profile target failed: {t__test_profile.error}",
        )

    try:
        build_behavioral_coverage(
            env.gateway,
            env.snapshot,
            llm_runner=None,
        )

        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.behavioral_coverage
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return BehavioralCoverageResult(
            success=True,
            table_counts={"analytics.behavioral_coverage": row_count},
        )

    except Exception as exc:
        log.exception("Behavioral coverage computation failed")
        return BehavioralCoverageResult(
            success=False,
            table_counts={},
            error=str(exc),
        )


@tag(domain="analytics", target="behavioral_coverage", node_type="materialize")
def t__behavioral_coverage(
    env: BuildEnv,
    graph: TargetGraph,
    t__behavioral_coverage__compute: BehavioralCoverageResult,
) -> TargetRunRecord:
    """Materialize behavioral coverage target using executor template.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__behavioral_coverage__compute
        Computed behavioral coverage result.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, "behavioral_coverage", t__behavioral_coverage__compute)


__all__ = [
    "BehavioralCoverageResult",
    "CoverageTestEdgesResult",
    "t__behavioral_coverage",
    "t__behavioral_coverage__compute",
    "t__coverage_functions",
    "t__coverage_functions__compute",
    "t__coverage_test_edges",
    "t__coverage_test_edges__compute",
]
