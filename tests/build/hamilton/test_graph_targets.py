"""Tests for graph_targets.py graphs module.

This module validates that the consolidated graph targets in
``codeintel.build.hamilton.native.graphs.graph_targets`` work correctly
with the tool-target finalize helpers.

Tests cover:
- goids target (GOID extraction)
- symbol_uses target (symbol use edge extraction)
- graph_metrics target (graph-derived analytics)
- graph_validation target (integrity checks)
"""

from __future__ import annotations

from dataclasses import replace

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.graphs.graph_targets import (
    GOIDS_CROSSWALK_TABLE_KEY,
    GOIDS_GOIDS_TABLE_KEY,
    GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
    GRAPH_METRICS_MODULES_TABLE_KEY,
    GRAPH_STATS_TABLE_KEY,
    GRAPH_VALIDATION_TABLE_KEY,
    SYMBOL_USE_EDGES_TABLE_KEY,
    GoidsToolOutput,
    GraphMetricsToolOutput,
    GraphValidationToolOutput,
    SymbolUsesToolOutput,
    t__goids,
    t__goids__ingest,
    t__graph_metrics,
    t__graph_metrics__ingest,
    t__graph_validation,
    t__graph_validation__ingest,
    t__symbol_uses,
    t__symbol_uses__ingest,
)
from codeintel.build.hamilton.native.patterns import ToolFinalizeContext
from codeintel.core.execution.materialization import MaterializationStatus
from tests._helpers.assertions import (
    assert_record_row_counts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.catalog import build_catalog, make_target_descriptor
from tests._helpers.contracts import contract_for_keys
from tests._helpers.harnesses.graph_harness import GraphTargetHarness

# Test constants to avoid magic numbers
MAX_GOID_COUNT = 50
MAX_SYMBOL_USES_COUNT = 100
MAX_GRAPH_METRICS_COUNT = 25


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


def _make_graph() -> DagCatalog:
    """Create a minimal catalog for graph targets.

    Returns
    -------
    DagCatalog
        Catalog with graph targets registered.
    """
    return build_catalog(
        targets=(
            make_target_descriptor(
                name="goids",
                module="graphs",
                contract=contract_for_keys(("core.goids", "core.goid_crosswalk")),
            ),
            make_target_descriptor(
                name="symbol_uses",
                module="graphs",
                contract=contract_for_keys(("graph.symbol_use_edges",)),
            ),
            make_target_descriptor(
                name="graph_metrics",
                module="graphs",
                contract=contract_for_keys(
                    (
                        "analytics.graph_metrics_functions",
                        "analytics.graph_metrics_modules",
                    )
                ),
            ),
            make_target_descriptor(
                name="graph_validation",
                module="graphs",
                contract=contract_for_keys(("analytics.graph_validation",)),
            ),
        )
    )


def _make_materialization(
    table_key: str,
    row_count: int,
    *,
    status: MaterializationStatus = "succeeded",
    error: str | None = None,
) -> MaterializationResult:
    return MaterializationResult(
        status=status,
        table_key=table_key,
        row_count=row_count,
        duration_ms=0.0,
        input_hash="test",
        error=error,
    )


def _make_graph_metrics_materializations(
    *,
    functions: int,
    functions_ext: int,
    modules: int,
    modules_ext: int,
    stats: int,
) -> dict[str, MaterializationResult]:
    return {
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY: _make_materialization(
            GRAPH_METRICS_FUNCTIONS_TABLE_KEY, functions
        ),
        GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY: _make_materialization(
            GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY, functions_ext
        ),
        GRAPH_METRICS_MODULES_TABLE_KEY: _make_materialization(
            GRAPH_METRICS_MODULES_TABLE_KEY, modules
        ),
        GRAPH_METRICS_MODULES_EXT_TABLE_KEY: _make_materialization(
            GRAPH_METRICS_MODULES_EXT_TABLE_KEY, modules_ext
        ),
        GRAPH_STATS_TABLE_KEY: _make_materialization(GRAPH_STATS_TABLE_KEY, stats),
    }


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

    goid_rows = tuple((idx,) for idx in range(MAX_GOID_COUNT))
    crosswalk_rows = tuple((idx,) for idx in range(MAX_GOID_COUNT))
    tool_output = GoidsToolOutput(
        result=ExecutionResult.ok(
            table_counts={
                GOIDS_GOIDS_TABLE_KEY: MAX_GOID_COUNT,
                GOIDS_CROSSWALK_TABLE_KEY: MAX_GOID_COUNT,
            }
        ),
        goid_rows=goid_rows,
        crosswalk_rows=crosswalk_rows,
    )
    ingest = t__goids__ingest(tool_output)
    materializations = {
        GOIDS_GOIDS_TABLE_KEY: _make_materialization(GOIDS_GOIDS_TABLE_KEY, MAX_GOID_COUNT),
        GOIDS_CROSSWALK_TABLE_KEY: _make_materialization(GOIDS_CROSSWALK_TABLE_KEY, MAX_GOID_COUNT),
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="goids",
    )

    record = t__goids(finalize_context, tool_output, ingest, materializations)

    assert_target_ok(record)
    assert_record_row_counts(record, {GOIDS_GOIDS_TABLE_KEY: MAX_GOID_COUNT})


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

    tool_output = GoidsToolOutput(result=ExecutionResult.failed("Upstream modules failed"))
    ingest = t__goids__ingest(tool_output)
    materializations = {
        GOIDS_GOIDS_TABLE_KEY: _make_materialization(GOIDS_GOIDS_TABLE_KEY, 0),
        GOIDS_CROSSWALK_TABLE_KEY: _make_materialization(GOIDS_CROSSWALK_TABLE_KEY, 0),
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="goids",
    )

    record = t__goids(finalize_context, tool_output, ingest, materializations)

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

    edge_rows = tuple((idx,) for idx in range(MAX_SYMBOL_USES_COUNT))
    tool_output = SymbolUsesToolOutput(
        result=ExecutionResult.ok(table_counts={SYMBOL_USE_EDGES_TABLE_KEY: MAX_SYMBOL_USES_COUNT}),
        edge_rows=edge_rows,
    )
    ingest = t__symbol_uses__ingest(tool_output)
    materializations = {
        SYMBOL_USE_EDGES_TABLE_KEY: _make_materialization(
            SYMBOL_USE_EDGES_TABLE_KEY, MAX_SYMBOL_USES_COUNT
        )
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="symbol_uses",
    )

    record = t__symbol_uses(finalize_context, tool_output, ingest, materializations)

    assert_target_ok(record)
    assert_record_row_counts(record, {SYMBOL_USE_EDGES_TABLE_KEY: MAX_SYMBOL_USES_COUNT})


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

    tool_output = SymbolUsesToolOutput(result=ExecutionResult.failed("Upstream scip failed"))
    ingest = t__symbol_uses__ingest(tool_output)
    materializations = {
        SYMBOL_USE_EDGES_TABLE_KEY: _make_materialization(SYMBOL_USE_EDGES_TABLE_KEY, 0)
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="symbol_uses",
    )

    record = t__symbol_uses(finalize_context, tool_output, ingest, materializations)

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

    functions_rows = tuple((idx,) for idx in range(MAX_GRAPH_METRICS_COUNT))
    modules_rows = tuple((idx,) for idx in range(MAX_GRAPH_METRICS_COUNT))
    tool_output = GraphMetricsToolOutput(
        result=ExecutionResult.ok(
            table_counts={
                GRAPH_METRICS_FUNCTIONS_TABLE_KEY: MAX_GRAPH_METRICS_COUNT,
                GRAPH_METRICS_MODULES_TABLE_KEY: MAX_GRAPH_METRICS_COUNT,
                GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY: 0,
                GRAPH_METRICS_MODULES_EXT_TABLE_KEY: 0,
                GRAPH_STATS_TABLE_KEY: 0,
            }
        ),
        functions_rows=functions_rows,
        modules_rows=modules_rows,
    )
    ingest = t__graph_metrics__ingest(tool_output)
    materializations = _make_graph_metrics_materializations(
        functions=MAX_GRAPH_METRICS_COUNT,
        functions_ext=0,
        modules=MAX_GRAPH_METRICS_COUNT,
        modules_ext=0,
        stats=0,
    )
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="graph_metrics",
    )

    record = t__graph_metrics(finalize_context, tool_output, ingest, materializations)

    assert_target_ok(record)
    assert_record_row_counts(
        record,
        {GRAPH_METRICS_FUNCTIONS_TABLE_KEY: MAX_GRAPH_METRICS_COUNT},
    )


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

    tool_output = GraphMetricsToolOutput(
        result=ExecutionResult.failed("Upstream call_graph failed")
    )
    ingest = t__graph_metrics__ingest(tool_output)
    materializations = _make_graph_metrics_materializations(
        functions=0,
        functions_ext=0,
        modules=0,
        modules_ext=0,
        stats=0,
    )
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="graph_metrics",
    )

    record = t__graph_metrics(finalize_context, tool_output, ingest, materializations)

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

    tool_output = GraphValidationToolOutput(
        result=ExecutionResult.ok(table_counts={GRAPH_VALIDATION_TABLE_KEY: 0}),
        rows=(),
    )
    ingest = t__graph_validation__ingest(tool_output)
    materializations = {
        GRAPH_VALIDATION_TABLE_KEY: _make_materialization(GRAPH_VALIDATION_TABLE_KEY, 0)
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="graph_validation",
    )

    record = t__graph_validation(finalize_context, tool_output, ingest, materializations)

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

    tool_output = GraphValidationToolOutput(
        result=ExecutionResult.failed("Upstream call_graph failed")
    )
    ingest = t__graph_validation__ingest(tool_output)
    materializations = {
        GRAPH_VALIDATION_TABLE_KEY: _make_materialization(GRAPH_VALIDATION_TABLE_KEY, 0)
    }
    finalize_context = ToolFinalizeContext(
        env=env,
        catalog=graph,
        target_name="graph_validation",
    )

    record = t__graph_validation(finalize_context, tool_output, ingest, materializations)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        "Upstream call_graph failed" in (record.error or ""),
        message="Error message should be propagated",
    )
