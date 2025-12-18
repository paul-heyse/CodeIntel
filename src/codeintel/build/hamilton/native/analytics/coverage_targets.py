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

import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    source,
    step,
    value,
)

from codeintel.analytics.compute.coverage.compute import (
    aggregate_coverage_lines,
    enrich_coverage_results,
    filter_coverage_lines_for_snapshot,
    filter_goids_for_snapshot,
    join_goids_with_coverage_lines,
)
from codeintel.analytics.testing import compute_test_coverage_edges
from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize, tag_tool
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.storage.queries.safe import count_rows_for_snapshot

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)

COVERAGE_FUNCTIONS_TARGET_NAME = "coverage_functions"
COVERAGE_TEST_EDGES_TARGET_NAME = "coverage_test_edges"
BEHAVIORAL_COVERAGE_TARGET_NAME = "behavioral_coverage"

COVERAGE_FUNCTIONS_TABLE_KEY = "analytics.coverage_functions"
TEST_COVERAGE_EDGES_TABLE_KEY = "analytics.test_coverage_edges"
BEHAVIORAL_COVERAGE_TABLE_KEY = "analytics.behavioral_coverage"

TARGET_SPECS = (
    make_output_target(
        name=COVERAGE_FUNCTIONS_TARGET_NAME,
        module="analytics",
        description="Per-function coverage aggregation.",
        options=TargetSpecOptions(table_keys=(COVERAGE_FUNCTIONS_TABLE_KEY,)),
    ),
    make_output_target(
        name=COVERAGE_TEST_EDGES_TARGET_NAME,
        module="analytics",
        description="Test-to-function coverage edges.",
        options=TargetSpecOptions(table_keys=(TEST_COVERAGE_EDGES_TABLE_KEY,)),
    ),
    make_output_target(
        name=BEHAVIORAL_COVERAGE_TARGET_NAME,
        module="analytics",
        description="Behavioral coverage tagging from test patterns.",
        options=TargetSpecOptions(table_keys=(BEHAVIORAL_COVERAGE_TABLE_KEY,)),
    ),
)


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


@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node(COVERAGE_FUNCTIONS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(COVERAGE_FUNCTIONS_TARGET_NAME),
    table_key=value(COVERAGE_FUNCTIONS_TABLE_KEY),
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
@tag_compute(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
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


@tag_materialize(domain="analytics", target=COVERAGE_FUNCTIONS_TARGET_NAME)
def t__coverage_functions(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__coverage_functions: MaterializationMetadata,
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
        target_name=COVERAGE_FUNCTIONS_TARGET_NAME,
        expected_table_key=COVERAGE_FUNCTIONS_TABLE_KEY,
        materialization=m__analytics__coverage_functions,
    )


# -----------------------------------------------------------------------------
# Coverage test edges (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def t__coverage_test_edges__compute(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> ExecutionResult:
    """Compute test-to-function coverage edges.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    ExecutionResult
        Result indicating success or failure with table counts.
    """
    if t__goids.status != "succeeded":
        return ExecutionResult.failed(f"Upstream goids target failed: {t__goids.error}")

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

        return ExecutionResult.ok(table_counts={TEST_COVERAGE_EDGES_TABLE_KEY: 0})

    except Exception as exc:
        log.exception("Coverage test edges computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def t__coverage_test_edges(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_test_edges__compute: ExecutionResult,
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
    return executor_materialize(
        env,
        graph,
        COVERAGE_TEST_EDGES_TARGET_NAME,
        t__coverage_test_edges__compute,
    )


# -----------------------------------------------------------------------------
# Behavioral coverage (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def t__behavioral_coverage__compute(
    env: BuildEnv,
    t__test_profile: TargetRunRecord,
) -> ExecutionResult:
    """Assign heuristic behavior tags to tests.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__test_profile
        Upstream test_profile target result (for dependency).

    Returns
    -------
    ExecutionResult
        Result indicating success or failure with table counts.
    """
    if t__test_profile.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream test_profile target failed: {t__test_profile.error}"
        )

    try:
        build_behavioral_coverage(
            env.gateway,
            env.snapshot,
            llm_runner=None,
        )

        row_count = count_rows_for_snapshot(
            env.gateway.con,
            BEHAVIORAL_COVERAGE_TABLE_KEY,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        return ExecutionResult.ok(table_counts={BEHAVIORAL_COVERAGE_TABLE_KEY: row_count})

    except Exception as exc:
        log.exception("Behavioral coverage computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def t__behavioral_coverage(
    env: BuildEnv,
    graph: TargetGraph,
    t__behavioral_coverage__compute: ExecutionResult,
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
    return executor_materialize(
        env,
        graph,
        BEHAVIORAL_COVERAGE_TARGET_NAME,
        t__behavioral_coverage__compute,
    )


__all__ = [
    "t__behavioral_coverage",
    "t__behavioral_coverage__compute",
    "t__coverage_functions",
    "t__coverage_functions__compute",
    "t__coverage_test_edges",
    "t__coverage_test_edges__compute",
]
