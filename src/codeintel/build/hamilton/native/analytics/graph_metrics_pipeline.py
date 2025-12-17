"""Consolidated Hamilton implementation for graph metrics targets.

This module provides Hamilton native nodes for three graph metrics targets:
- `subsystem_graph_metrics`: Metrics for subsystem coupling and centrality
- `symbol_graph_metrics`: Metrics for symbol usage patterns
- `test_graph_metrics`: Metrics from test-function bipartite graph

All targets share common patterns for graph runtime resolution and
result handling, consolidated here to reduce code duplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

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
from codeintel.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
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
# Subsystem graph metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemGraphMetricsResult:
    """Result from subsystem graph metrics computation."""

    success: bool
    row_count: int = 0
    error: str | None = None


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="compute")
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
        Result indicating success or failure with row count.
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

        return SubsystemGraphMetricsResult(success=True, row_count=row_count)

    except Exception as exc:
        log.exception("Subsystem graph metrics computation failed")
        return SubsystemGraphMetricsResult(success=False, error=str(exc))


@tag(domain="analytics", target="subsystem_graph_metrics", node_type="materialize")
def t__subsystem_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_graph_metrics__compute: SubsystemGraphMetricsResult,
) -> TargetRunRecord:
    """Materialize subsystem graph metrics target.

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
    executor = NativeTargetExecutor.for_target(env, graph, "subsystem_graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__subsystem_graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(
                t__subsystem_graph_metrics__compute.error or "Subsystem graph metrics failed"
            )
        )

    def compute() -> dict[str, int]:
        return {"analytics.subsystem_graph_metrics": t__subsystem_graph_metrics__compute.row_count}

    return executor.execute(compute)


# -----------------------------------------------------------------------------
# Symbol graph metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolGraphMetricsResult:
    """Result from symbol graph metrics computation."""

    success: bool
    row_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(domain="analytics", target="symbol_graph_metrics", node_type="compute")
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
        Result indicating success or failure with row counts.
    """
    if t__symbol_uses.status != "succeeded":
        return SymbolGraphMetricsResult(
            success=False,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    row_counts: dict[str, int] = {}

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
            row_counts["analytics.symbol_graph_metrics_modules"] = int(row[0]) if row else 0
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
            row_counts["analytics.symbol_graph_metrics_functions"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Symbol graph metrics (functions) failed: %s", exc)

        log.info("Symbol graph metrics completed: %s", row_counts)
        return SymbolGraphMetricsResult(success=True, row_counts=row_counts)

    except Exception as exc:
        log.exception("Symbol graph metrics computation failed")
        return SymbolGraphMetricsResult(success=False, error=str(exc))


@tag(domain="analytics", target="symbol_graph_metrics", node_type="materialize")
def t__symbol_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__symbol_graph_metrics__compute: SymbolGraphMetricsResult,
) -> TargetRunRecord:
    """Materialize symbol graph metrics target.

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
    executor = NativeTargetExecutor.for_target(env, graph, "symbol_graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__symbol_graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(t__symbol_graph_metrics__compute.error or "Symbol graph metrics failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__symbol_graph_metrics__compute.row_counts)

    return executor.execute(compute)


# -----------------------------------------------------------------------------
# Test graph metrics
# -----------------------------------------------------------------------------


@tag(domain="analytics", target="test_graph_metrics", node_type="compute")
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
    "SubsystemGraphMetricsResult",
    "SymbolGraphMetricsResult",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
    "test_graph_metrics__functions_rows",
    "test_graph_metrics__tests_rows",
]
