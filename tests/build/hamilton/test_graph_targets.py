"""Tests for graph_targets.py graphs module.

This module validates that the consolidated graph targets in
``codeintel.build.hamilton.native.graphs.graph_targets`` work correctly
with the executor_materialize helper for Pattern D targets.

Tests cover:
- goids target (GOID extraction)
- symbol_uses target (symbol use edge extraction)
- graph_metrics target (graph-derived analytics)
- graph_validation target (integrity checks)
"""

from __future__ import annotations

from dataclasses import replace

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.graphs.graph_targets import (
    GraphMetricsComputeResult,
    GraphMetricsMaterializations,
    GraphValidationResult,
    t__goids,
    t__graph_metrics,
    t__graph_validation,
    t__symbol_uses,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import (
    assert_record_row_counts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.contracts import contract_for_keys
from tests._helpers.harnesses.graph_harness import GraphTargetHarness

# Test constants to avoid magic numbers
MAX_GOID_COUNT = 50
MAX_SYMBOL_USES_COUNT = 100
MAX_GRAPH_METRICS_COUNT = 25
MAX_GRAPH_VALIDATION_ERRORS = 10


def _make_env(harness: GraphTargetHarness) -> BuildEnv:
    """Create a BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    return replace(
        harness.harness.build_env(),
        force_targets=frozenset(
            {
                "goids",
                "symbol_uses",
                "graph_metrics",
                "graph_validation",
            }
        ),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for graph targets.

    Returns
    -------
    TargetGraph
        Target graph with graph targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="goids",
            module="graphs",
            contract=contract_for_keys(("core.goids", "core.goid_crosswalk")),
        )
    )
    graph.register(
        OutputTarget(
            name="symbol_uses",
            module="graphs",
            contract=contract_for_keys(("graph.symbol_use_edges",)),
        )
    )
    graph.register(
        OutputTarget(
            name="graph_metrics",
            module="graphs",
            contract=contract_for_keys(
                (
                    "analytics.graph_metrics_functions",
                    "analytics.graph_metrics_modules",
                )
            ),
        )
    )
    graph.register(
        OutputTarget(
            name="graph_validation",
            module="graphs",
            contract=contract_for_keys(("analytics.graph_validation",)),
        )
    )
    return graph


def _make_materialization(
    table_key: str,
    row_count: int,
    *,
    status: str = "succeeded",
    error: str | None = None,
) -> MaterializationMetadata:
    return {
        "status": status,
        "table_key": table_key,
        "row_count": row_count,
        "duration_ms": 0.0,
        "input_hash": "test",
        "error": error,
    }


def _make_graph_metrics_materializations(
    *,
    functions: int,
    functions_ext: int,
    modules: int,
    modules_ext: int,
    stats: int,
) -> GraphMetricsMaterializations:
    return GraphMetricsMaterializations(
        functions=_make_materialization("analytics.graph_metrics_functions", functions),
        functions_ext=_make_materialization("analytics.graph_metrics_functions_ext", functions_ext),
        modules=_make_materialization("analytics.graph_metrics_modules", modules),
        modules_ext=_make_materialization("analytics.graph_metrics_modules_ext", modules_ext),
        graph_stats=_make_materialization("analytics.graph_stats", stats),
    )


# ---------------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------------


def test_execution_result_success() -> None:
    """Verify ExecutionResult for success case."""
    result = ExecutionResult.ok(
        table_counts={
            "analytics.graph_metrics_functions": MAX_GRAPH_METRICS_COUNT,
            "analytics.graph_metrics_modules": MAX_GRAPH_METRICS_COUNT,
        }
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(
        result.table_counts["analytics.graph_metrics_functions"],
        MAX_GRAPH_METRICS_COUNT,
    )
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream call_graph failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream call_graph failed")


# ---------------------------------------------------------------------------
# GraphValidationResult Tests
# ---------------------------------------------------------------------------


def test_graph_validation_result_success() -> None:
    """Verify GraphValidationResult dataclass for success case (no errors)."""
    result = GraphValidationResult(
        success=True,
        error_count=0,
        issues=[],
        table_counts={"analytics.graph_validation": 0},
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.error_count, 0)
    expect_equal(len(result.issues), 0)
    expect_equal(result.error, None)


def test_graph_validation_result_with_errors() -> None:
    """Verify GraphValidationResult dataclass when validation finds issues."""
    validation_errors = [
        "Found 5 call graph edges with orphan caller GOIDs",
        "Found 3 import edges with missing source modules",
    ]
    result = GraphValidationResult(
        success=False,
        error_count=len(validation_errors),
        issues=validation_errors,
        table_counts={"analytics.graph_validation": len(validation_errors)},
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error_count, len(validation_errors))
    expect_equal(len(result.issues), len(validation_errors))


def test_graph_validation_result_fatal_failure() -> None:
    """Verify GraphValidationResult dataclass for fatal error case."""
    result = GraphValidationResult(
        success=False,
        table_counts={},
        error="Upstream call_graph target failed",
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream call_graph target failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests - goids
# ---------------------------------------------------------------------------


def test_goids_materialize_success(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__goids returns success record.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.ok(
        table_counts={
            "core.goids": MAX_GOID_COUNT,
            "core.goid_crosswalk": MAX_GOID_COUNT,
        }
    )

    record = t__goids(env, graph, compute_result)

    assert_target_ok(record)
    assert_record_row_counts(record, {"core.goids": MAX_GOID_COUNT})


def test_goids_materialize_failure(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__goids returns failure record when compute fails.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.failed("Upstream modules failed")

    record = t__goids(env, graph, compute_result)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream modules failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - symbol_uses
# ---------------------------------------------------------------------------


def test_symbol_uses_materialize_success(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__symbol_uses returns success record.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.ok(
        table_counts={"graph.symbol_use_edges": MAX_SYMBOL_USES_COUNT}
    )

    record = t__symbol_uses(env, graph, compute_result)

    assert_target_ok(record)
    assert_record_row_counts(record, {"graph.symbol_use_edges": MAX_SYMBOL_USES_COUNT})


def test_symbol_uses_materialize_failure(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__symbol_uses returns failure record when compute fails.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.failed("Upstream scip failed")

    record = t__symbol_uses(env, graph, compute_result)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream scip failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - graph_metrics
# ---------------------------------------------------------------------------


def test_graph_metrics_materialize_success(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__graph_metrics returns success record.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = GraphMetricsComputeResult(
        metrics=None,
        functions_ext_rows=None,
        modules_ext_rows=None,
        graph_stats_rows=None,
    )
    materializations = _make_graph_metrics_materializations(
        functions=MAX_GRAPH_METRICS_COUNT,
        functions_ext=0,
        modules=MAX_GRAPH_METRICS_COUNT,
        modules_ext=0,
        stats=0,
    )

    record = t__graph_metrics(env, graph, compute_result, materializations)

    assert_target_ok(record)
    assert_record_row_counts(record, {"analytics.graph_metrics_functions": MAX_GRAPH_METRICS_COUNT})


def test_graph_metrics_materialize_failure(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__graph_metrics returns failure record when compute fails.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = GraphMetricsComputeResult(
        metrics=None,
        functions_ext_rows=None,
        modules_ext_rows=None,
        graph_stats_rows=None,
        error="Upstream call_graph failed",
    )
    materializations = _make_graph_metrics_materializations(
        functions=0,
        functions_ext=0,
        modules=0,
        modules_ext=0,
        stats=0,
    )

    record = t__graph_metrics(env, graph, compute_result, materializations)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream call_graph failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - graph_validation
# ---------------------------------------------------------------------------


def test_graph_validation_materialize_success(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__graph_validation returns success record.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = GraphValidationResult(
        success=True,
        error_count=0,
        issues=[],
        table_counts={"analytics.graph_validation": 0},
    )

    record = t__graph_validation(env, graph, compute_result)

    assert_target_ok(record)


def test_graph_validation_materialize_failure(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Verify t__graph_validation returns failure record when validation fails.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    env = _make_env(graph_target_harness)
    graph = _make_graph()

    compute_result = GraphValidationResult(
        success=False,
        error_count=2,
        issues=["Error 1", "Error 2"],
        table_counts={"analytics.graph_validation": 2},
    )

    record = t__graph_validation(env, graph, compute_result)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Error 1" in (record.error or "") or "Error 2" in (record.error or ""),
        message="Error messages should be propagated",
    )
