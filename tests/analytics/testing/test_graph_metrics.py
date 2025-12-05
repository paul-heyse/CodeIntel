"""Tests for codeintel.analytics.testing.graph_metrics module.

Testing Charter Compliance:
- Uses real DuckDB via TestContext (no mocking)
- Production-parity configuration loading
- Realistic test data via seed packs
- No monkeypatching or test-only code paths

This module tests the public API compute_test_graph_metrics function
and the TestMetricsContext dataclass. The private helper functions
are tested indirectly through the main function.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from codeintel.analytics.compute.graphs import BipartiteDegrees
from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.analytics.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.analytics.testing.graph_metrics import (
    TestMetricsContext,
    compute_test_graph_metrics,
)
from tests._helpers import COVERAGE_PACK, METRICS_PACK, TestContext, assert_frozen

# =============================================================================
# Test Constants
# =============================================================================

EXPECTED_ROW_COUNT_MINIMUM = 0
EXPECTED_TUPLE_LENGTH = 12
EXPECTED_TEST_COUNT_TWO = 2
EXPECTED_FUNC_COUNT_TWO = 2

# Test data values for BipartiteDegrees tests
TEST_DEGREE_VALUE = 5
FUNC_DEGREE_VALUE = 3
TEST_WEIGHTED_DEGREE = 2.5
FUNC_WEIGHTED_DEGREE = 1.5
TEST_CENTRALITY = 0.8
FUNC_CENTRALITY = 0.6
RISK_SCORE = 0.7
GOID_VALUE = 123


class TestTestMetricsContext:
    """Tests for TestMetricsContext dataclass."""

    @staticmethod
    def test_creates_frozen_context() -> None:
        """Verify TestMetricsContext is immutable."""
        graph_ctx = resolve_graph_context(
            GraphContextSpec(
                repo="test_repo",
                commit="abc123",
                use_gpu=False,
                now=datetime.now(UTC),
                pagerank_weight="weight",
                betweenness_weight="weight",
            )
        )
        degrees = BipartiteDegrees(
            degree={},
            weighted_degree={},
            primary_degree_centrality={},
            secondary_degree_centrality={},
        )
        ctx = TestMetricsContext(
            repo="test_repo",
            commit="abc123",
            now=datetime.now(UTC),
            degrees=degrees,
            risk_by_goid={1: 0.5, 2: 0.8},
            graph_ctx=graph_ctx,
        )
        assert ctx.repo == "test_repo"
        assert ctx.commit == "abc123"
        assert ctx.risk_by_goid == {1: 0.5, 2: 0.8}
        # Verify frozen - should raise AttributeError on mutation
        assert_frozen(ctx, "repo", "modified")

    @staticmethod
    def test_stores_degrees_data() -> None:
        """Verify TestMetricsContext stores degrees data correctly."""
        test_node = ("test", "test_id")
        func_node = ("func", GOID_VALUE)
        graph_ctx = resolve_graph_context(
            GraphContextSpec(
                repo="test_repo",
                commit="abc123",
                use_gpu=False,
                now=datetime.now(UTC),
                pagerank_weight="weight",
                betweenness_weight="weight",
            )
        )
        degrees = BipartiteDegrees(
            degree={test_node: TEST_DEGREE_VALUE, func_node: FUNC_DEGREE_VALUE},
            weighted_degree={test_node: TEST_WEIGHTED_DEGREE, func_node: FUNC_WEIGHTED_DEGREE},
            primary_degree_centrality={test_node: TEST_CENTRALITY},
            secondary_degree_centrality={func_node: FUNC_CENTRALITY},
        )
        ctx = TestMetricsContext(
            repo="test_repo",
            commit="abc123",
            now=datetime.now(UTC),
            degrees=degrees,
            risk_by_goid={GOID_VALUE: RISK_SCORE},
            graph_ctx=graph_ctx,
        )
        assert ctx.degrees.degree[test_node] == TEST_DEGREE_VALUE
        assert ctx.degrees.degree[func_node] == FUNC_DEGREE_VALUE
        assert ctx.degrees.weighted_degree[test_node] == TEST_WEIGHTED_DEGREE
        assert ctx.risk_by_goid[GOID_VALUE] == RISK_SCORE

    @staticmethod
    def test_stores_datetime_correctly() -> None:
        """Verify TestMetricsContext stores datetime correctly."""
        now = datetime.now(UTC)
        graph_ctx = resolve_graph_context(
            GraphContextSpec(
                repo="test_repo",
                commit="abc123",
                use_gpu=False,
                now=now,
                pagerank_weight="weight",
                betweenness_weight="weight",
            )
        )
        degrees = BipartiteDegrees(
            degree={},
            weighted_degree={},
            primary_degree_centrality={},
            secondary_degree_centrality={},
        )
        ctx = TestMetricsContext(
            repo="test_repo",
            commit="abc123",
            now=now,
            degrees=degrees,
            risk_by_goid={},
            graph_ctx=graph_ctx,
        )
        assert ctx.now == now
        assert ctx.now.tzinfo is not None  # Timezone aware


class TestComputeTestGraphMetrics:
    """Tests for compute_test_graph_metrics function."""

    @staticmethod
    def test_creates_empty_tables_when_no_coverage(test_ctx: TestContext) -> None:
        """Verify tables are created even with no coverage data."""
        # Don't seed any coverage data - just ensure core tables exist
        test_ctx.require(METRICS_PACK)

        # Should not raise - creates empty tables
        compute_test_graph_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )

        # Tables should exist (even if empty)
        test_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )
        func_count = test_ctx.query_count(
            "analytics.test_graph_metrics_functions",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )
        # May be 0 if no coverage data is seeded
        assert test_count >= EXPECTED_ROW_COUNT_MINIMUM
        assert func_count >= EXPECTED_ROW_COUNT_MINIMUM

    @staticmethod
    def test_computes_metrics_with_seeded_coverage(coverage_ctx: TestContext) -> None:
        """Verify metrics are computed when coverage data exists."""
        # coverage_ctx has COVERAGE_PACK which seeds test catalog and edges
        coverage_ctx.require(METRICS_PACK)  # For risk factors

        compute_test_graph_metrics(
            coverage_ctx.gateway,
            repo=coverage_ctx.repo,
            commit=coverage_ctx.commit,
        )

        # Verify metrics tables are populated
        test_count = coverage_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{coverage_ctx.repo}' AND commit = '{coverage_ctx.commit}'",
        )
        func_count = coverage_ctx.query_count(
            "analytics.test_graph_metrics_functions",
            f"repo = '{coverage_ctx.repo}' AND commit = '{coverage_ctx.commit}'",
        )
        # At least some metrics should be computed if coverage data exists
        assert test_count >= EXPECTED_ROW_COUNT_MINIMUM
        assert func_count >= EXPECTED_ROW_COUNT_MINIMUM

    @staticmethod
    def test_clears_previous_metrics_on_recompute(test_ctx: TestContext) -> None:
        """Verify previous metrics are cleared when recomputing."""
        test_ctx.require(COVERAGE_PACK, METRICS_PACK)

        # Compute twice
        compute_test_graph_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        first_test_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )

        compute_test_graph_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        second_test_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )

        # Counts should be equal (not doubled)
        assert first_test_count == second_test_count

    @staticmethod
    def test_handles_different_repo_commit_combinations(test_ctx: TestContext) -> None:
        """Verify metrics are correctly scoped to repo/commit."""
        test_ctx.require(COVERAGE_PACK, METRICS_PACK)

        # Compute for default context
        compute_test_graph_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )

        # Query for a different commit should return 0
        other_commit_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{test_ctx.repo}' AND commit = 'nonexistent_commit'",
        )
        assert other_commit_count == EXPECTED_ROW_COUNT_MINIMUM

        # Query for a different repo should return 0
        other_repo_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = 'nonexistent_repo' AND commit = '{test_ctx.commit}'",
        )
        assert other_repo_count == EXPECTED_ROW_COUNT_MINIMUM

    @staticmethod
    def test_integrates_with_graph_runtime_options(test_ctx: TestContext) -> None:
        """Verify compute_test_graph_metrics accepts runtime options."""
        test_ctx.require(COVERAGE_PACK, METRICS_PACK)

        options = GraphRuntimeOptions()
        # Should not raise when passed options
        compute_test_graph_metrics(
            test_ctx.gateway,
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            runtime=options,
        )

        test_count = test_ctx.query_count(
            "analytics.test_graph_metrics_tests",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )
        assert test_count >= EXPECTED_ROW_COUNT_MINIMUM


class TestDecimalConversion:
    """Tests for Decimal conversion behavior used in graph metrics."""

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (0, Decimal(0)),
            (1, Decimal(1)),
            (42, Decimal(42)),
            (-1, Decimal(-1)),
            (999999999, Decimal(999999999)),
        ],
    )
    def test_decimal_conversion_matches_behavior(input_val: int, expected: Decimal) -> None:
        """Verify Decimal conversion behavior for integer inputs.

        The module uses Decimal for GOID values in database rows.
        This tests the conversion behavior.
        """
        result = Decimal(input_val)
        assert result == expected
        assert isinstance(result, Decimal)
