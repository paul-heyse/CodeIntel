"""Demonstration of hexagonal test helper architecture.

This module shows how to use the new hexagonal test infrastructure to
write cleaner, more maintainable tests with minimal boilerplate.

Compare these tests with the original test_graph_metrics_smoke.py to see
how the new architecture reduces setup code while maintaining the same
production-parity testing principles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graphs import compute_graph_metrics
from codeintel.config import ConfigBuilder
from tests._helpers import (
    COVERAGE_PACK,
    GRAPH_PACK,
    METRICS_PACK,
    TestContext,
    TestScenario,
)
from tests._helpers.context import SeedPack
from tests._helpers.seeds.core import GOID_FUNC_A, GOID_FUNC_B

# Constants for expected counts (avoids magic number lint)
EXPECTED_GRAPH_PACK_GOIDS = 4
EXPECTED_QUERY_ROW_COUNT = 2
METRICS_PACK_FUNCTION_COUNT = 4


# =============================================================================
# Graph Metrics Tests Using Hexagonal Fixtures
# =============================================================================


def test_compute_graph_metrics_with_seeded_data(graph_ctx: TestContext) -> None:
    """Verify graph metrics populate with pre-seeded graph data.

    This is equivalent to test_compute_graph_metrics_with_small_graph
    but uses the seeded data from GRAPH_PACK instead of manual insertion.
    """
    cfg = ConfigBuilder.from_snapshot(
        repo=graph_ctx.repo,
        commit=graph_ctx.commit,
        repo_root=graph_ctx.repo_root,
    ).graph_metrics()

    compute_graph_metrics(graph_ctx.gateway, cfg)

    # Use the TestContext query helper for assertions
    row_count = graph_ctx.query_count(
        "analytics.graph_metrics_functions",
        f"repo = '{graph_ctx.repo}' AND commit = '{graph_ctx.commit}'",
    )
    # GRAPH_PACK seeds 4 GOIDs with call graph nodes
    expected_rows = EXPECTED_GRAPH_PACK_GOIDS
    assert row_count == expected_rows, f"Expected {expected_rows} function metric rows, got {row_count}"


def test_query_helper_returns_typed_rows(graph_ctx: TestContext) -> None:
    """Demonstrate QueryRow access patterns.

    The TestContext.query() method returns QueryRow objects that support
    both index and attribute access.
    """
    rows = graph_ctx.query(
        "SELECT goid_h128, qualname FROM core.goids WHERE goid_h128 IN (?, ?)",
        [GOID_FUNC_A, GOID_FUNC_B],
    )

    # Both access patterns work
    assert len(rows) == EXPECTED_QUERY_ROW_COUNT
    for row in rows:
        # Index access - goid_h128 is stored as DECIMAL(38,0), returns Decimal
        assert row[0] is not None
        # Attribute access
        assert isinstance(row.qualname, str)
        # Dict conversion
        row_dict = row.as_dict()
        assert "goid_h128" in row_dict


# =============================================================================
# Scenario Builder Tests
# =============================================================================


def test_custom_scenario_with_multiple_packs(tmp_path: Path) -> None:
    """Create a custom scenario combining multiple seed packs."""
    ctx = (
        TestScenario.minimal()
        .with_seeds(GRAPH_PACK, COVERAGE_PACK)
        .with_sample_files()
        .build(tmp_path)
    )

    try:
        # Verify all expected data is present
        assert ctx.query_count("core.goids") > 0
        assert ctx.query_count("graph.call_graph_edges") > 0
        assert ctx.query_count("analytics.test_catalog") > 0

        # Verify sample files were written
        assert (ctx.repo_root / "pkg" / "mod_a.py").exists()
    finally:
        ctx.close()


def test_scenario_with_custom_repo_commit(tmp_path: Path) -> None:
    """Customize repository and commit identifiers."""
    ctx = (
        TestScenario.minimal()
        .with_repo("custom/repo")
        .with_commit("abc123")
        .build(tmp_path)
    )

    try:
        assert ctx.repo == "custom/repo"
        assert ctx.commit == "abc123"

        # Seeded data uses the custom identifiers
        rows = ctx.query("SELECT DISTINCT repo FROM core.modules")
        assert rows[0][0] == "custom/repo"
    finally:
        ctx.close()


# =============================================================================
# Seed Pack Composition Tests
# =============================================================================


def test_require_packs_idempotently(test_ctx: TestContext) -> None:
    """Seed packs can be required multiple times safely.

    The TestContext tracks which packs have been applied and
    skips re-application.
    """
    # First application
    test_ctx.require(GRAPH_PACK)
    initial_count = test_ctx.query_count("graph.call_graph_edges")

    # Second application is idempotent
    test_ctx.require(GRAPH_PACK)
    second_count = test_ctx.query_count("graph.call_graph_edges")

    assert initial_count == second_count


def test_pack_dependencies_resolved(test_ctx: TestContext) -> None:
    """Dependent packs are automatically applied.

    GRAPH_PACK depends on CORE_PACK, so requiring GRAPH_PACK
    automatically seeds core data first.
    """
    # Only require GRAPH_PACK
    test_ctx.require(GRAPH_PACK)

    # Core data was also seeded due to dependency
    assert test_ctx.query_count("core.modules") > 0
    assert test_ctx.query_count("core.goids") > 0


def test_full_stack_provides_all_data(full_ctx: TestContext) -> None:
    """The full_ctx fixture provides comprehensive test data.

    This is useful for integration tests that need multiple data types.
    """
    # Core data
    assert full_ctx.query_count("core.modules") > 0
    assert full_ctx.query_count("core.goids") > 0

    # Graph data
    assert full_ctx.query_count("graph.call_graph_edges") > 0
    assert full_ctx.query_count("graph.import_graph_edges") > 0

    # Coverage data
    assert full_ctx.query_count("analytics.test_catalog") > 0
    assert full_ctx.query_count("analytics.test_coverage_edges") > 0

    # Metrics data
    assert full_ctx.query_count("analytics.function_metrics") > 0
    assert full_ctx.query_count("analytics.goid_risk_factors") > 0


# =============================================================================
# Metrics Data Access Tests
# =============================================================================


def test_metrics_ctx_provides_function_metrics(metrics_ctx: TestContext) -> None:
    """The metrics_ctx fixture provides function metrics data."""
    rows = metrics_ctx.query(
        """
        SELECT function_goid_h128, qualname, cyclomatic_complexity
        FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [metrics_ctx.repo, metrics_ctx.commit],
    )

    assert len(rows) == METRICS_PACK_FUNCTION_COUNT  # METRICS_PACK seeds 4 functions
    for row in rows:
        assert isinstance(row.cyclomatic_complexity, int)
        assert row.cyclomatic_complexity >= 1


def test_risk_factors_correlate_with_metrics(metrics_ctx: TestContext) -> None:
    """Risk factors reference the same GOIDs as function metrics."""
    metric_goids = {
        row[0]
        for row in metrics_ctx.query(
            "SELECT DISTINCT function_goid_h128 FROM analytics.function_metrics"
        )
    }
    risk_goids = {
        row[0]
        for row in metrics_ctx.query(
            "SELECT DISTINCT function_goid_h128 FROM analytics.goid_risk_factors"
        )
    }

    # Both tables have data for the same GOIDs
    assert metric_goids == risk_goids


# =============================================================================
# Parametrized Pack Tests
# =============================================================================


@pytest.mark.parametrize(
    ("pack", "expected_table"),
    [
        (GRAPH_PACK, "graph.call_graph_edges"),
        (COVERAGE_PACK, "analytics.test_catalog"),
        (METRICS_PACK, "analytics.function_metrics"),
    ],
)
def test_pack_seeds_expected_table(
    test_ctx: TestContext,
    pack: SeedPack,
    expected_table: str,
) -> None:
    """Verify each pack seeds its expected primary table.

    Parameters
    ----------
    test_ctx
        Base test context fixture.
    pack
        Seed pack to apply.
    expected_table
        Table that should have data after pack application.
    """
    test_ctx.require(pack)
    count = test_ctx.query_count(expected_table)
    assert count > 0, f"Expected {expected_table} to have rows after {pack.name}"
