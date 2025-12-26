"""Tests for coverage_targets.py analytics module."""

from __future__ import annotations

from dataclasses import replace

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.analytics.coverage_targets import (
    BehavioralCoverageComputeResult,
    CoverageTestEdgesComputeResult,
    t__behavioral_coverage,
    t__coverage_test_edges,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import (
    assert_record_row_counts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.contracts import contract_for_keys
from tests._helpers.harnesses.analytics_harness import AnalyticsTargetHarness


def _make_env(harness: AnalyticsTargetHarness) -> BuildEnv:
    """Create a BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    return replace(
        harness.harness.build_env(),
        force_targets=frozenset({"coverage_test_edges", "behavioral_coverage"}),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for coverage targets.

    Returns
    -------
    TargetGraph
        Target graph with coverage targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="coverage_test_edges",
            module="analytics",
            contract=contract_for_keys(("analytics.test_coverage_edges",)),
        )
    )
    graph.register(
        OutputTarget(
            name="behavioral_coverage",
            module="analytics",
            contract=contract_for_keys(("analytics.behavioral_coverage",)),
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


# ---------------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------------


def test_execution_result_success() -> None:
    """Verify ExecutionResult for success case."""
    result = ExecutionResult.ok(table_counts={"analytics.test_coverage_edges": 100})
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.table_counts["analytics.test_coverage_edges"], 100)
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests
# ---------------------------------------------------------------------------


def test_coverage_test_edges_materialize_success(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__coverage_test_edges returns success record.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = CoverageTestEdgesComputeResult(rows=[])
    materialization = _make_materialization("analytics.test_coverage_edges", 25)

    record = t__coverage_test_edges(
        env,
        graph,
        compute_result,
        {"analytics.test_coverage_edges": materialization},
    )

    expected_count = 25
    assert_target_ok(record)
    assert_record_row_counts(record, {"analytics.test_coverage_edges": expected_count})


def test_coverage_test_edges_materialize_failure(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__coverage_test_edges returns failure record when compute fails.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = CoverageTestEdgesComputeResult(rows=None, error="Upstream goids failed")
    materialization = _make_materialization(
        "analytics.test_coverage_edges",
        0,
        status="failed",
        error="Upstream goids failed",
    )

    record = t__coverage_test_edges(
        env,
        graph,
        compute_result,
        {"analytics.test_coverage_edges": materialization},
    )

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream goids failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_behavioral_coverage_materialize_success(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__behavioral_coverage returns success record.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = BehavioralCoverageComputeResult(rows=[])
    materialization = _make_materialization("analytics.behavioral_coverage", 15)

    record = t__behavioral_coverage(
        env,
        graph,
        compute_result,
        {"analytics.behavioral_coverage": materialization},
    )

    expected_count = 15
    assert_target_ok(record)
    assert_record_row_counts(record, {"analytics.behavioral_coverage": expected_count})


def test_behavioral_coverage_materialize_failure(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__behavioral_coverage returns failure record when compute fails.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = BehavioralCoverageComputeResult(rows=None, error="Test profile failed")
    materialization = _make_materialization(
        "analytics.behavioral_coverage",
        0,
        status="failed",
        error="Test profile failed",
    )

    record = t__behavioral_coverage(
        env,
        graph,
        compute_result,
        {"analytics.behavioral_coverage": materialization},
    )

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Test profile failed" in (record.error or ""),
        message="Error message should be propagated",
    )
