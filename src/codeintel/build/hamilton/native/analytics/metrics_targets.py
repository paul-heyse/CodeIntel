"""Consolidated Hamilton implementation for metrics-related analytics targets.

This module consolidates metrics analytics targets using Phase 1 templates:

History Targets (Pattern B - Rows):
- ``function_history``: Per-function creation/modification/churn metrics
- ``history_timeseries``: Multi-commit timeseries analytics

Graph Metrics Targets (Pattern D - Executor):
- ``subsystem_graph_metrics``: Metrics for subsystem coupling and centrality
- ``symbol_graph_metrics``: Metrics for symbol usage patterns
- ``subsystem_agreement``: Subsystem assignment agreement metrics

Multi-Table Target (Pattern C):
- ``test_graph_metrics``: Metrics from test-function bipartite graph

All Pattern D targets use the ``executor_materialize`` template for simplified
materialize nodes with ``NativeTargetExecutor`` pattern.
"""

from __future__ import annotations

import logging

from hamilton.function_modifiers import source, value

from codeintel.analytics.functions.function_history import build_function_history_rows
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize, tag_tool
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.graphs.runtime import GraphRuntime, resolve_graph_runtime
from codeintel.storage.queries.safe import count_rows_for_snapshot

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    GraphRuntime,
    TargetGraph,
    TargetRunRecord,
    TestGraphMetricsResult,
)

FUNCTION_HISTORY_TARGET_NAME = "function_history"
HISTORY_TIMESERIES_TARGET_NAME = "history_timeseries"
SUBSYSTEM_GRAPH_METRICS_TARGET_NAME = "subsystem_graph_metrics"
SYMBOL_GRAPH_METRICS_TARGET_NAME = "symbol_graph_metrics"
SUBSYSTEM_AGREEMENT_TARGET_NAME = "subsystem_agreement"
TEST_GRAPH_METRICS_TARGET_NAME = "test_graph_metrics"

FUNCTION_HISTORY_TABLE_KEY = "analytics.function_history"
HISTORY_TIMESERIES_TABLE_KEY = "analytics.history_timeseries"
SUBSYSTEM_GRAPH_METRICS_TABLE_KEY = "analytics.subsystem_graph_metrics"
SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.symbol_graph_metrics_modules"
SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.symbol_graph_metrics_functions"
SYMBOL_GRAPH_METRICS_TABLE_KEYS = (
    SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY,
    SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
)
SUBSYSTEM_AGREEMENT_TABLE_KEY = "analytics.subsystem_agreement"
TEST_GRAPH_METRICS_TESTS_TABLE_KEY = "analytics.test_graph_metrics_tests"
TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.test_graph_metrics_functions"
TEST_GRAPH_METRICS_TABLE_KEYS = (
    TEST_GRAPH_METRICS_TESTS_TABLE_KEY,
    TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
)

TARGET_SPECS = (
    make_output_target(
        name=FUNCTION_HISTORY_TARGET_NAME,
        module="analytics",
        description="Function git history and churn metrics.",
        options=TargetSpecOptions(
            table_keys=(FUNCTION_HISTORY_TABLE_KEY,),
            allow_declared_overrides=True,
        ),
    ),
    make_output_target(
        name=HISTORY_TIMESERIES_TARGET_NAME,
        module="analytics",
        description="Historical metrics timeseries for trending.",
        options=TargetSpecOptions(
            table_keys=(HISTORY_TIMESERIES_TABLE_KEY,),
            allow_declared_overrides=True,
        ),
    ),
    make_output_target(
        name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
        module="analytics",
        description="Graph metrics for subsystems.",
        options=TargetSpecOptions(
            table_keys=(SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,),
            allow_declared_overrides=True,
        ),
    ),
    make_output_target(
        name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
        module="analytics",
        description="Graph metrics from symbol usage patterns.",
        options=TargetSpecOptions(
            table_keys=SYMBOL_GRAPH_METRICS_TABLE_KEYS,
            allow_declared_overrides=True,
        ),
    ),
    make_output_target(
        name=SUBSYSTEM_AGREEMENT_TARGET_NAME,
        module="analytics",
        description="Subsystem vs import community agreement.",
        options=TargetSpecOptions(
            table_keys=(SUBSYSTEM_AGREEMENT_TABLE_KEY,),
            allow_declared_overrides=True,
        ),
    ),
    make_output_target(
        name=TEST_GRAPH_METRICS_TARGET_NAME,
        module="analytics",
        description="Graph metrics from test-function bipartite graph.",
        options=TargetSpecOptions(
            table_keys=TEST_GRAPH_METRICS_TABLE_KEYS,
            allow_declared_overrides=True,
        ),
    ),
)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def _get_graph_runtime(env: BuildEnv, *, target_name: str) -> GraphRuntime | None:
    """Resolve graph runtime from build environment.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    target_name
        Target name used to load runtime options from configuration.

    Returns
    -------
    GraphRuntime | None
        Resolved graph runtime, or None if resolution fails.
    """
    try:
        options = load_graph_runtime_options(env, target_name=target_name)
        return resolve_graph_runtime(env.gateway, env.snapshot, options)
    except (RuntimeError, ValueError) as exc:
        log.warning("Failed to resolve graph runtime: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Function history (Pattern B - Rows)
# -----------------------------------------------------------------------------


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_HISTORY_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_HISTORY_TARGET_NAME),
    table_key=value(FUNCTION_HISTORY_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FUNCTION_HISTORY_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
    target_="t__function_history__compute",
)
def t__function_history__compute(
    env: BuildEnv,
    graph: TargetGraph,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute function history metrics for all functions.

    This is a pure compute node with no side effects. It computes git history
    and churn metrics for each function and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for manifest-driven skip checks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples matching the function_history schema order, or None when skipped.

    Notes
    -----
    The metrics computed include:
    - Function creation and last modification dates
    - Commit count and author count
    - Lines added and deleted (churn)
    - Stability bucket classification
    """
    target = graph.get(FUNCTION_HISTORY_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, FUNCTION_HISTORY_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return build_function_history_rows(env.gateway, env.snapshot)


@tag_materialize(domain="analytics", target=FUNCTION_HISTORY_TARGET_NAME)
def t__function_history(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_history: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize function history table to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__function_history
        Materialization metadata for analytics.function_history.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=FUNCTION_HISTORY_TARGET_NAME,
        expected_table_key=FUNCTION_HISTORY_TABLE_KEY,
        materialization=m__analytics__function_history,
    )


# -----------------------------------------------------------------------------
# History timeseries (Pattern B - Rows)
# -----------------------------------------------------------------------------


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(HISTORY_TIMESERIES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(HISTORY_TIMESERIES_TARGET_NAME),
    table_key=value(HISTORY_TIMESERIES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(HISTORY_TIMESERIES_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
    target_="t__history_timeseries__compute",
)
def t__history_timeseries__compute(env: BuildEnv) -> tuple[tuple[object, ...], ...]:
    """Compute history timeseries metrics across commits.

    Full multi-commit functionality is not yet wired into ``BuildEnv``, so this
    node currently returns an empty result set and succeeds.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Empty row set until multi-commit configuration is available.
    """
    _ = env
    log.info(
        "history_timeseries: Multi-commit configuration not available via BuildEnv; "
        "returning empty result set."
    )
    return ()


@tag_materialize(domain="analytics", target=HISTORY_TIMESERIES_TARGET_NAME)
def t__history_timeseries(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__history_timeseries: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize history timeseries table to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__history_timeseries
        Materialization metadata for analytics.history_timeseries.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=HISTORY_TIMESERIES_TARGET_NAME,
        expected_table_key=HISTORY_TIMESERIES_TABLE_KEY,
        materialization=m__analytics__history_timeseries,
    )


# -----------------------------------------------------------------------------
# Subsystem graph metrics (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
def t__subsystem_graph_metrics__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> ExecutionResult:
    """Compute graph metrics for subsystems.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    ExecutionResult
        Result indicating success or failure with table counts.
    """
    if t__subsystems.status != "succeeded":
        return ExecutionResult.failed(f"Upstream subsystems target failed: {t__subsystems.error}")

    try:
        graph_runtime = _get_graph_runtime(env, target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
        log.info(
            "Computing subsystem graph metrics for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        compute_subsystem_graph_metrics(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            runtime=graph_runtime,
        )

        row_count = count_rows_for_snapshot(
            env.gateway.con,
            SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        return ExecutionResult.ok(table_counts={SUBSYSTEM_GRAPH_METRICS_TABLE_KEY: row_count})

    except Exception as exc:
        log.exception("Subsystem graph metrics computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="analytics", target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
def t__subsystem_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_graph_metrics__compute: ExecutionResult,
) -> TargetRunRecord:
    """Materialize subsystem graph metrics target using executor template.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__subsystem_graph_metrics__compute
        Computed subsystem graph metrics result.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(
        env,
        graph,
        SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
        t__subsystem_graph_metrics__compute,
    )


# -----------------------------------------------------------------------------
# Symbol graph metrics (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=SYMBOL_GRAPH_METRICS_TARGET_NAME)
def t__symbol_graph_metrics__compute(
    env: BuildEnv,
    t__symbol_uses: TargetRunRecord,
) -> ExecutionResult:
    """Compute graph metrics from symbol usage patterns.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__symbol_uses
        Upstream symbol_uses target result (for dependency).

    Returns
    -------
    ExecutionResult
        Result indicating success or failure with table counts.
    """
    if t__symbol_uses.status != "succeeded":
        return ExecutionResult.failed(f"Upstream symbol_uses target failed: {t__symbol_uses.error}")

    table_counts: dict[str, int] = {
        SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY: 0,
        SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY: 0,
    }
    errors: list[str] = []

    try:
        graph_runtime = _get_graph_runtime(env, target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME)
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        try:
            log.info("Computing symbol graph metrics (modules) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_modules(
                env.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            table_counts[SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY] = count_rows_for_snapshot(
                env.gateway.con,
                SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            errors.append(f"modules: {exc}")
            log.warning("Symbol graph metrics (modules) failed: %s", exc)

        try:
            log.info("Computing symbol graph metrics (functions) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_functions(
                env.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            table_counts[SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY] = count_rows_for_snapshot(
                env.gateway.con,
                SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            errors.append(f"functions: {exc}")
            log.warning("Symbol graph metrics (functions) failed: %s", exc)

        log.info("Symbol graph metrics completed: %s", table_counts)
        if errors:
            return ExecutionResult.failed("; ".join(errors), table_counts=table_counts)
        return ExecutionResult.ok(table_counts=table_counts)

    except Exception as exc:
        log.exception("Symbol graph metrics computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="analytics", target=SYMBOL_GRAPH_METRICS_TARGET_NAME)
def t__symbol_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__symbol_graph_metrics__compute: ExecutionResult,
) -> TargetRunRecord:
    """Materialize symbol graph metrics target using executor template.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__symbol_graph_metrics__compute
        Computed symbol graph metrics result.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(
        env,
        graph,
        SYMBOL_GRAPH_METRICS_TARGET_NAME,
        t__symbol_graph_metrics__compute,
    )


# -----------------------------------------------------------------------------
# Subsystem agreement (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=SUBSYSTEM_AGREEMENT_TARGET_NAME)
def t__subsystem_agreement__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> ExecutionResult:
    """Compare subsystem assignments with import community labels.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    ExecutionResult
        Status indicator, table counts, and optional error message.
    """
    if t__subsystems.status != "succeeded":
        return ExecutionResult.failed(f"Upstream subsystems target failed: {t__subsystems.error}")

    try:
        log.info(
            "Computing subsystem agreement for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        compute_subsystem_agreement(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        row_count = count_rows_for_snapshot(
            env.gateway.con,
            SUBSYSTEM_AGREEMENT_TABLE_KEY,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        return ExecutionResult.ok(table_counts={SUBSYSTEM_AGREEMENT_TABLE_KEY: row_count})
    except Exception as exc:
        log.exception("Subsystem agreement computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="analytics", target=SUBSYSTEM_AGREEMENT_TARGET_NAME)
def t__subsystem_agreement(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_agreement__compute: ExecutionResult,
) -> TargetRunRecord:
    """Materialize subsystem agreement target using executor template.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__subsystem_agreement__compute
        Computed subsystem agreement result.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return executor_materialize(
        env,
        graph,
        SUBSYSTEM_AGREEMENT_TARGET_NAME,
        t__subsystem_agreement__compute,
    )


# -----------------------------------------------------------------------------
# Test graph metrics (Pattern C - Multi-Table)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=TEST_GRAPH_METRICS_TARGET_NAME)
def t__test_graph_metrics__compute(
    env: BuildEnv,
    graph: TargetGraph,
) -> TestGraphMetricsResult | None:
    """Compute test graph metrics for all tests and functions.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for skip check.

    Returns
    -------
    TestGraphMetricsResult | None
        Container with rows for test and function metrics tables, or None if skipped.
    """
    target = graph.get(TEST_GRAPH_METRICS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, TEST_GRAPH_METRICS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    runtime = _get_graph_runtime(env, target_name=TEST_GRAPH_METRICS_TARGET_NAME)
    if runtime is None:
        return TestGraphMetricsResult(test_rows=(), function_rows=())
    return compute_test_graph_metrics_pure(env.gateway, env.snapshot, runtime)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(TEST_GRAPH_METRICS_TESTS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TEST_GRAPH_METRICS_TARGET_NAME),
    table_key=value(TEST_GRAPH_METRICS_TESTS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(TEST_GRAPH_METRICS_TESTS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
    target_="test_graph_metrics__tests_rows",
)
def test_graph_metrics__tests_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_tests.

    Parameters
    ----------
    t__test_graph_metrics__compute
        Computed test graph metrics result.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for the test metrics table, or None if compute was skipped.
    """
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.test_rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TEST_GRAPH_METRICS_TARGET_NAME),
    table_key=value(TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
    target_="test_graph_metrics__functions_rows",
)
def test_graph_metrics__functions_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_functions.

    Parameters
    ----------
    t__test_graph_metrics__compute
        Computed test graph metrics result.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for the function metrics table, or None if compute was skipped.
    """
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.function_rows)


@tag_materialize(domain="analytics", target=TEST_GRAPH_METRICS_TARGET_NAME)
def t__test_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__test_graph_metrics_tests: MaterializationMetadata,
    m__analytics__test_graph_metrics_functions: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize both test graph metrics tables to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__test_graph_metrics_tests
        Materialization metadata for tests table.
    m__analytics__test_graph_metrics_functions
        Materialization metadata for functions table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=TEST_GRAPH_METRICS_TARGET_NAME,
        materializations={
            TEST_GRAPH_METRICS_TESTS_TABLE_KEY: m__analytics__test_graph_metrics_tests,
            TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY: m__analytics__test_graph_metrics_functions,
        },
    )


__all__ = [
    "t__function_history",
    "t__function_history__compute",
    "t__history_timeseries",
    "t__history_timeseries__compute",
    "t__subsystem_agreement",
    "t__subsystem_agreement__compute",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
    "test_graph_metrics__functions_rows",
    "test_graph_metrics__tests_rows",
]
