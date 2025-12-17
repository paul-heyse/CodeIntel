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
from dataclasses import dataclass, field
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.history.history_timeseries import HISTORY_TIMESERIES_COLS
from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import make_output_target
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    GraphRuntime,
    TargetGraph,
    TargetRunRecord,
    TestGraphMetricsResult,
)

TARGET_SPECS = (
    make_output_target(
        name="function_history",
        module="analytics",
        description="Function git history and churn metrics.",
        table_keys=("analytics.function_history",),
    ),
    make_output_target(
        name="history_timeseries",
        module="analytics",
        description="Historical metrics timeseries for trending.",
        table_keys=("analytics.history_timeseries",),
    ),
    make_output_target(
        name="subsystem_graph_metrics",
        module="analytics",
        description="Graph metrics for subsystems.",
        table_keys=("analytics.subsystem_graph_metrics",),
    ),
    make_output_target(
        name="symbol_graph_metrics",
        module="analytics",
        description="Graph metrics from symbol usage patterns.",
        table_keys=(
            "analytics.symbol_graph_metrics_modules",
            "analytics.symbol_graph_metrics_functions",
        ),
    ),
    make_output_target(
        name="subsystem_agreement",
        module="analytics",
        description="Subsystem vs import community agreement.",
        table_keys=("analytics.subsystem_agreement",),
    ),
    make_output_target(
        name="test_graph_metrics",
        module="analytics",
        description="Graph metrics from test-function bipartite graph.",
        table_keys=(
            "analytics.test_graph_metrics_tests",
            "analytics.test_graph_metrics_functions",
        ),
    ),
)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def _get_graph_runtime(env: BuildEnv) -> GraphRuntime | None:
    """Resolve graph runtime from build environment.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    GraphRuntime | None
        Resolved graph runtime, or None if resolution fails.
    """
    try:
        return resolve_graph_runtime(env.gateway, env.snapshot, GraphRuntimeOptions())
    except (RuntimeError, ValueError) as exc:
        log.warning("Failed to resolve graph runtime: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Result dataclasses (adapted for executor_materialize template)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemGraphMetricsResult:
    """Result from subsystem graph metrics computation.

    Follows ComputeResult protocol for executor_materialize template.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class SymbolGraphMetricsResult:
    """Result from symbol graph metrics computation.

    Follows ComputeResult protocol for executor_materialize template.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class SubsystemAgreementResult:
    """Result from subsystem agreement computation.

    Follows ComputeResult protocol for executor_materialize template.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# -----------------------------------------------------------------------------
# Function history (Pattern B - Rows)
# -----------------------------------------------------------------------------


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.function_history"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("function_history"),
    table_key=value("analytics.function_history"),
    columns=value(tuple(FUNCTION_HISTORY_COLS)),
)
@tag(
    domain="analytics",
    target="function_history",
    node_type="compute",
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
        Row tuples matching FUNCTION_HISTORY_COLS schema, or None when skipped.

    Notes
    -----
    The metrics computed include:
    - Function creation and last modification dates
    - Commit count and author count
    - Lines added and deleted (churn)
    - Stability bucket classification
    """
    target = graph.get("function_history")
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
    return build_function_history_rows(env.gateway, env.snapshot)


@tag(domain="analytics", target="function_history", node_type="materialize")
def t__function_history(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_history: dict[str, Any],
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
        target_name="function_history",
        expected_table_key="analytics.function_history",
        materialization=m__analytics__function_history,
    )


# -----------------------------------------------------------------------------
# History timeseries (Pattern B - Rows)
# -----------------------------------------------------------------------------


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.history_timeseries"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("history_timeseries"),
    table_key=value("analytics.history_timeseries"),
    columns=value(tuple(HISTORY_TIMESERIES_COLS)),
)
@tag(
    domain="analytics",
    target="history_timeseries",
    node_type="compute",
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


@tag(domain="analytics", target="history_timeseries", node_type="materialize")
def t__history_timeseries(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__history_timeseries: dict[str, Any],
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
        target_name="history_timeseries",
        expected_table_key="analytics.history_timeseries",
        materialization=m__analytics__history_timeseries,
    )


# -----------------------------------------------------------------------------
# Subsystem graph metrics (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="tool")
def t__subsystem_graph_metrics__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> SubsystemGraphMetricsResult:
    """Compute graph metrics for subsystems.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    SubsystemGraphMetricsResult
        Result indicating success or failure with table counts.
    """
    if t__subsystems.status != "succeeded":
        return SubsystemGraphMetricsResult(
            success=False,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

    try:
        graph_runtime = _get_graph_runtime(env)
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

        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_graph_metrics
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return SubsystemGraphMetricsResult(
            success=True,
            table_counts={"analytics.subsystem_graph_metrics": row_count},
        )

    except Exception as exc:
        log.exception("Subsystem graph metrics computation failed")
        return SubsystemGraphMetricsResult(success=False, error=str(exc))


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="materialize")
def t__subsystem_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_graph_metrics__compute: SubsystemGraphMetricsResult,
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
        env, graph, "subsystem_graph_metrics", t__subsystem_graph_metrics__compute
    )


# -----------------------------------------------------------------------------
# Symbol graph metrics (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="symbol_graph_metrics", node_type="tool")
def t__symbol_graph_metrics__compute(
    env: BuildEnv,
    t__symbol_uses: TargetRunRecord,
) -> SymbolGraphMetricsResult:
    """Compute graph metrics from symbol usage patterns.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__symbol_uses
        Upstream symbol_uses target result (for dependency).

    Returns
    -------
    SymbolGraphMetricsResult
        Result indicating success or failure with table counts.
    """
    if t__symbol_uses.status != "succeeded":
        return SymbolGraphMetricsResult(
            success=False,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    table_counts: dict[str, int] = {}

    try:
        graph_runtime = _get_graph_runtime(env)
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
            row = env.gateway.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            table_counts["analytics.symbol_graph_metrics_modules"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Symbol graph metrics (modules) failed: %s", exc)

        try:
            log.info("Computing symbol graph metrics (functions) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_functions(
                env.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            row = env.gateway.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            table_counts["analytics.symbol_graph_metrics_functions"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Symbol graph metrics (functions) failed: %s", exc)

        log.info("Symbol graph metrics completed: %s", table_counts)
        return SymbolGraphMetricsResult(success=True, table_counts=table_counts)

    except Exception as exc:
        log.exception("Symbol graph metrics computation failed")
        return SymbolGraphMetricsResult(success=False, error=str(exc))


@tag(domain="analytics", target="symbol_graph_metrics", node_type="materialize")
def t__symbol_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__symbol_graph_metrics__compute: SymbolGraphMetricsResult,
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
        env, graph, "symbol_graph_metrics", t__symbol_graph_metrics__compute
    )


# -----------------------------------------------------------------------------
# Subsystem agreement (Pattern D - Executor)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="subsystem_agreement", node_type="tool")
def t__subsystem_agreement__compute(
    env: BuildEnv,
    t__subsystems: TargetRunRecord,
) -> SubsystemAgreementResult:
    """Compare subsystem assignments with import community labels.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__subsystems
        Upstream subsystems target result (for dependency).

    Returns
    -------
    SubsystemAgreementResult
        Status indicator, table counts, and optional error message.
    """
    if t__subsystems.status != "succeeded":
        return SubsystemAgreementResult(
            success=False,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

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

        row = env.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_agreement
            WHERE repo = ? AND commit = ?
            """,
            [env.snapshot.repo, env.snapshot.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return SubsystemAgreementResult(
            success=True,
            table_counts={"analytics.subsystem_agreement": row_count},
        )
    except Exception as exc:
        log.exception("Subsystem agreement computation failed")
        return SubsystemAgreementResult(success=False, error=str(exc))


@tag(domain="analytics", target="subsystem_agreement", node_type="materialize")
def t__subsystem_agreement(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_agreement__compute: SubsystemAgreementResult,
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
    return executor_materialize(env, graph, "subsystem_agreement", t__subsystem_agreement__compute)


# -----------------------------------------------------------------------------
# Test graph metrics (Pattern C - Multi-Table)
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="test_graph_metrics", node_type="tool")
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
    target = graph.get("test_graph_metrics")
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

    runtime = _get_graph_runtime(env)
    if runtime is None:
        return TestGraphMetricsResult(test_rows=(), function_rows=())
    return compute_test_graph_metrics_pure(env.gateway, env.snapshot, runtime)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_graph_metrics_tests"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_graph_metrics"),
    table_key=value("analytics.test_graph_metrics_tests"),
    columns=value(tuple(TEST_GRAPH_METRICS_TESTS_COLS)),
)
@tag(
    domain="analytics",
    target="test_graph_metrics",
    node_type="compute",
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


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_graph_metrics_functions"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_graph_metrics"),
    table_key=value("analytics.test_graph_metrics_functions"),
    columns=value(tuple(TEST_GRAPH_METRICS_FUNCTIONS_COLS)),
)
@tag(
    domain="analytics",
    target="test_graph_metrics",
    node_type="compute",
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


@tag(domain="analytics", target="test_graph_metrics", node_type="materialize")
def t__test_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__test_graph_metrics_tests: dict[str, Any],
    m__analytics__test_graph_metrics_functions: dict[str, Any],
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
        target_name="test_graph_metrics",
        materializations={
            "analytics.test_graph_metrics_tests": m__analytics__test_graph_metrics_tests,
            "analytics.test_graph_metrics_functions": m__analytics__test_graph_metrics_functions,
        },
    )


__all__ = [
    "SubsystemAgreementResult",
    "SubsystemGraphMetricsResult",
    "SymbolGraphMetricsResult",
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
