"""Tests for metrics_targets.py analytics module.

This module validates that the consolidated metrics targets in
``codeintel.build.hamilton.native.analytics.metrics_targets`` work correctly
with the executor_materialize helper for Pattern D targets.
"""

from __future__ import annotations

from dataclasses import replace

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.analytics.metrics_targets import (
    SubsystemAgreementComputeResult,
    SubsystemGraphMetricsComputeResult,
    SymbolGraphMetricsComputeResult,
    t__subsystem_agreement,
    t__subsystem_graph_metrics,
    t__symbol_graph_metrics,
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
        force_targets=frozenset(
            {"subsystem_graph_metrics", "symbol_graph_metrics", "subsystem_agreement"}
        ),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for metrics targets.

    Returns
    -------
    TargetGraph
        Target graph with metrics targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="subsystem_graph_metrics",
            module="analytics",
            contract=contract_for_keys(("analytics.subsystem_graph_metrics",)),
        )
    )
    graph.register(
        OutputTarget(
            name="symbol_graph_metrics",
            module="analytics",
            contract=contract_for_keys(
                (
                    "analytics.symbol_graph_metrics_modules",
                    "analytics.symbol_graph_metrics_functions",
                )
            ),
        )
    )
    graph.register(
        OutputTarget(
            name="subsystem_agreement",
            module="analytics",
            contract=contract_for_keys(("analytics.subsystem_agreement",)),
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
    result = ExecutionResult.ok(table_counts={"analytics.subsystem_graph_metrics": 100})
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.table_counts["analytics.subsystem_graph_metrics"], 100)
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream subsystems failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream subsystems failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests
# ---------------------------------------------------------------------------


def test_subsystem_graph_metrics_materialize_success(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__subsystem_graph_metrics returns success record.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SubsystemGraphMetricsComputeResult(rows=[])
    materialization = _make_materialization("analytics.subsystem_graph_metrics", 25)

    record = t__subsystem_graph_metrics(env, graph, compute_result, materialization)

    expected_count = 25
    assert_target_ok(record)
    assert_record_row_counts(record, {"analytics.subsystem_graph_metrics": expected_count})


def test_subsystem_graph_metrics_materialize_failure(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__subsystem_graph_metrics returns failure record when compute fails.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SubsystemGraphMetricsComputeResult(
        rows=None,
        error="Upstream subsystems failed",
    )
    materialization = _make_materialization(
        "analytics.subsystem_graph_metrics",
        0,
        status="failed",
        error="Upstream subsystems failed",
    )

    record = t__subsystem_graph_metrics(env, graph, compute_result, materialization)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream subsystems failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_symbol_graph_metrics_materialize_success(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__symbol_graph_metrics returns success record.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SymbolGraphMetricsComputeResult(module_rows=[], function_rows=[])
    modules_meta = _make_materialization("analytics.symbol_graph_metrics_modules", 10)
    functions_meta = _make_materialization("analytics.symbol_graph_metrics_functions", 50)

    record = t__symbol_graph_metrics(env, graph, compute_result, modules_meta, functions_meta)

    assert_target_ok(record)


def test_symbol_graph_metrics_materialize_failure(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__symbol_graph_metrics returns failure record when compute fails.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SymbolGraphMetricsComputeResult(
        module_rows=None,
        function_rows=None,
        error="Upstream symbol_uses failed",
    )
    modules_meta = _make_materialization(
        "analytics.symbol_graph_metrics_modules",
        0,
        status="failed",
        error="Upstream symbol_uses failed",
    )
    functions_meta = _make_materialization(
        "analytics.symbol_graph_metrics_functions",
        0,
        status="failed",
        error="Upstream symbol_uses failed",
    )

    record = t__symbol_graph_metrics(env, graph, compute_result, modules_meta, functions_meta)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream symbol_uses failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_subsystem_agreement_materialize_success(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__subsystem_agreement returns success record.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SubsystemAgreementComputeResult(rows=[])
    materialization = _make_materialization("analytics.subsystem_agreement", 15)

    record = t__subsystem_agreement(env, graph, compute_result, materialization)

    expected_count = 15
    assert_target_ok(record)
    assert_record_row_counts(record, {"analytics.subsystem_agreement": expected_count})


def test_subsystem_agreement_materialize_failure(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Verify t__subsystem_agreement returns failure record when compute fails.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    env = _make_env(analytics_target_harness)
    graph = _make_graph()

    compute_result = SubsystemAgreementComputeResult(
        rows=None,
        error="Upstream subsystems failed",
    )
    materialization = _make_materialization(
        "analytics.subsystem_agreement",
        0,
        status="failed",
        error="Upstream subsystems failed",
    )

    record = t__subsystem_agreement(env, graph, compute_result, materialization)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream subsystems failed" in (record.error or ""),
        message="Error message should be propagated",
    )
