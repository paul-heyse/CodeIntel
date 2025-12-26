"""Consolidated Hamilton implementation for coverage-related analytics targets.

This module consolidates coverage analytics targets using native materialization helpers:

- ``coverage_functions``: Per-function coverage aggregation (Ibis -> DuckDB)
- ``coverage_test_edges``: Test-to-function coverage edge computation (Rows)
- ``behavioral_coverage``: Heuristic behavior tag assignment for tests (Rows)

The coverage_functions target uses DAG-visible I/O via ``DuckDBIbisTableSaver``.
The other two targets use DAG-visible row materialization via ``DuckDBRowsSaver``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis.expr.types as ir
from hamilton.function_modifiers import check_output_custom, pipe_input, source, step

from codeintel.analytics.compute.coverage.compute import (
    aggregate_coverage_lines,
    enrich_coverage_results,
    filter_coverage_lines_for_snapshot,
    filter_goids_for_snapshot,
    join_goids_with_coverage_lines,
)
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.testing.behavioral.tags import build_behavior_rows
from codeintel.analytics.testing.coverage.edges import build_test_coverage_edges_rows
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
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
    TableSaveSpec,
    save_ibis_table,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord, options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsTestCoverageEdgesRow as TestCoverageEdgeRow,
    )

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)

COVERAGE_FUNCTIONS_TARGET_NAME = "coverage_functions"
COVERAGE_TEST_EDGES_TARGET_NAME = "coverage_test_edges"
BEHAVIORAL_COVERAGE_TARGET_NAME = "behavioral_coverage"

COVERAGE_FUNCTIONS_TABLE_KEY = "analytics.coverage_functions"
TEST_COVERAGE_EDGES_TABLE_KEY = "analytics.test_coverage_edges"
BEHAVIORAL_COVERAGE_TABLE_KEY = "analytics.behavioral_coverage"
COVERAGE_FUNCTIONS_TABLE_KEYS = (COVERAGE_FUNCTIONS_TABLE_KEY,)
TEST_COVERAGE_EDGES_TABLE_KEYS = (TEST_COVERAGE_EDGES_TABLE_KEY,)
BEHAVIORAL_COVERAGE_TABLE_KEYS = (BEHAVIORAL_COVERAGE_TABLE_KEY,)
COVERAGE_FUNCTIONS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
    hash_options_node="coverage_functions__hash_options",
)
COVERAGE_TEST_EDGES_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=COVERAGE_TEST_EDGES_TARGET_NAME,
    hash_options_node="coverage_test_edges__hash_options",
)
BEHAVIORAL_COVERAGE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=BEHAVIORAL_COVERAGE_TARGET_NAME,
    hash_options_node="behavioral_coverage__hash_options",
)


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for coverage nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@tag_helper(domain="analytics", target=COVERAGE_FUNCTIONS_TARGET_NAME)
def coverage_functions__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for coverage_functions execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, COVERAGE_FUNCTIONS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def coverage_test_edges__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for coverage_test_edges execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, COVERAGE_TEST_EDGES_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def coverage_test_edges__skip(
    env: BuildEnv,
    graph: TargetGraph,
    coverage_test_edges__hash_options: InputHashOptions,
) -> bool:
    """Return True when coverage_test_edges should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        COVERAGE_TEST_EDGES_TARGET_NAME,
        hash_options=coverage_test_edges__hash_options,
    )
    return executor.should_skip()


@tag_helper(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def behavioral_coverage__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for behavioral_coverage execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, BEHAVIORAL_COVERAGE_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def behavioral_coverage__skip(
    env: BuildEnv,
    graph: TargetGraph,
    behavioral_coverage__hash_options: InputHashOptions,
) -> bool:
    """Return True when behavioral_coverage should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        BEHAVIORAL_COVERAGE_TARGET_NAME,
        hash_options=behavioral_coverage__hash_options,
    )
    return executor.should_skip()


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


@save_ibis_table(
    context=COVERAGE_FUNCTIONS_SAVE_CONTEXT,
    spec=IbisTableSaveSpec(table_key=COVERAGE_FUNCTIONS_TABLE_KEY),
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


@codeintel_target(domain="analytics", target=COVERAGE_FUNCTIONS_TARGET_NAME)
def t__coverage_functions(
    env: BuildEnv,
    graph: TargetGraph,
    coverage_functions__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Aggregate per-function coverage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    coverage_functions__table_materializations
        Materialization metadata for analytics.coverage_functions.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=COVERAGE_FUNCTIONS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=coverage_functions__table_materializations,
    )


coverage_functions__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
    table_keys=COVERAGE_FUNCTIONS_TABLE_KEYS,
)


# -----------------------------------------------------------------------------
# Coverage test edges (Rows)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageTestEdgesComputeResult:
    """Result from coverage test edge computation."""

    rows: list[TestCoverageEdgeRow] | None
    error: str | None = None


@tag_compute(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def t__coverage_test_edges__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__goids: TargetRunRecord,
    *,
    coverage_test_edges__skip: bool,
) -> CoverageTestEdgesComputeResult:
    """Compute test-to-function coverage edges.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__goids
        Upstream goids target result (for dependency).
    coverage_test_edges__skip
        Skip flag derived from manifest-based input hash evaluation.
    coverage_test_edges__skip
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    CoverageTestEdgesComputeResult
        Row results and optional error.
    """
    if t__goids.status != "succeeded":
        return CoverageTestEdgesComputeResult(
            rows=None,
            error=f"Upstream goids target failed: {t__goids.error}",
        )
    if coverage_test_edges__skip:
        return CoverageTestEdgesComputeResult(rows=None)

    registry = build_registry(
        gateway=gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        try:
            catalog = registry.require(CatalogProvider).get()
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        rows = build_test_coverage_edges_rows(
            gateway,
            env.snapshot,
            catalog_provider=catalog,
        )

        return CoverageTestEdgesComputeResult(rows=rows)

    except Exception as exc:
        log.exception("Coverage test edges computation failed")
        return CoverageTestEdgesComputeResult(rows=None, error=str(exc))


@save_rows(
    context=COVERAGE_TEST_EDGES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TEST_COVERAGE_EDGES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=COVERAGE_TEST_EDGES_TARGET_NAME,
    target_="coverage_test_edges__rows",
)
def coverage_test_edges__rows(
    t__coverage_test_edges__compute: CoverageTestEdgesComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_coverage_edges table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__coverage_test_edges__compute.rows is None:
        return None
    return tuple(
        row_to_tuple(TEST_COVERAGE_EDGES_TABLE_KEY, row)
        for row in t__coverage_test_edges__compute.rows
    )


@codeintel_target(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
def t__coverage_test_edges(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_test_edges__compute: CoverageTestEdgesComputeResult,
    coverage_test_edges__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Compute test-to-function coverage edges.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__coverage_test_edges__compute
        Computed coverage edges result.
    coverage_test_edges__table_materializations
        Materialization metadata for analytics.test_coverage_edges.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__coverage_test_edges__compute.error:
        options_hash = options_hash_for_target(env, COVERAGE_TEST_EDGES_TARGET_NAME)
        return TargetRunRecord(
            target=COVERAGE_TEST_EDGES_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__coverage_test_edges__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=COVERAGE_TEST_EDGES_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=coverage_test_edges__table_materializations,
    )


coverage_test_edges__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=COVERAGE_TEST_EDGES_TARGET_NAME,
    table_keys=TEST_COVERAGE_EDGES_TABLE_KEYS,
)


# -----------------------------------------------------------------------------
# Behavioral coverage (Rows)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BehavioralCoverageComputeResult:
    """Result from behavioral coverage row construction."""

    rows: list[tuple[object, ...]] | None
    error: str | None = None


@tag_compute(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def t__behavioral_coverage__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__test_profile: TargetRunRecord,
    *,
    behavioral_coverage__skip: bool,
) -> BehavioralCoverageComputeResult:
    """Assign heuristic behavior tags to tests.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__test_profile
        Upstream test_profile target result (for dependency).
    behavioral_coverage__skip
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    BehavioralCoverageComputeResult
        Row results and optional error.
    """
    if t__test_profile.status != "succeeded":
        return BehavioralCoverageComputeResult(
            rows=None,
            error=f"Upstream test_profile target failed: {t__test_profile.error}",
        )
    if behavioral_coverage__skip:
        return BehavioralCoverageComputeResult(rows=None)

    try:
        rows = build_behavior_rows(
            gateway,
            env.snapshot,
            llm_runner=None,
        )
        return BehavioralCoverageComputeResult(rows=rows)

    except Exception as exc:
        log.exception("Behavioral coverage computation failed")
        return BehavioralCoverageComputeResult(rows=None, error=str(exc))


@save_rows(
    context=BEHAVIORAL_COVERAGE_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=BEHAVIORAL_COVERAGE_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=BEHAVIORAL_COVERAGE_TARGET_NAME,
    target_="behavioral_coverage__rows",
)
def behavioral_coverage__rows(
    t__behavioral_coverage__compute: BehavioralCoverageComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.behavioral_coverage table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__behavioral_coverage__compute.rows is None:
        return None
    return tuple(t__behavioral_coverage__compute.rows)


@codeintel_target(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def t__behavioral_coverage(
    env: BuildEnv,
    graph: TargetGraph,
    t__behavioral_coverage__compute: BehavioralCoverageComputeResult,
    behavioral_coverage__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Tag behavioral coverage from test patterns.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__behavioral_coverage__compute
        Computed behavioral coverage result.
    behavioral_coverage__table_materializations
        Materialization metadata for analytics.behavioral_coverage.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__behavioral_coverage__compute.error:
        options_hash = options_hash_for_target(env, BEHAVIORAL_COVERAGE_TARGET_NAME)
        return TargetRunRecord(
            target=BEHAVIORAL_COVERAGE_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__behavioral_coverage__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=BEHAVIORAL_COVERAGE_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=behavioral_coverage__table_materializations,
    )


behavioral_coverage__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=BEHAVIORAL_COVERAGE_TARGET_NAME,
    table_keys=BEHAVIORAL_COVERAGE_TABLE_KEYS,
)


__all__ = [
    "BehavioralCoverageComputeResult",
    "CoverageTestEdgesComputeResult",
    "behavioral_coverage__hash_options",
    "behavioral_coverage__rows",
    "behavioral_coverage__skip",
    "behavioral_coverage__table_materializations",
    "coverage_functions__hash_options",
    "coverage_functions__table_materializations",
    "coverage_test_edges__hash_options",
    "coverage_test_edges__rows",
    "coverage_test_edges__skip",
    "coverage_test_edges__table_materializations",
    "t__behavioral_coverage",
    "t__behavioral_coverage__compute",
    "t__coverage_functions",
    "t__coverage_functions__compute",
    "t__coverage_test_edges",
    "t__coverage_test_edges__compute",
]
