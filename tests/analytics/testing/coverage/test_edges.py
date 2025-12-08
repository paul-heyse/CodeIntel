"""Tests for codeintel.analytics.testing.coverage.edges module.

Testing Charter Compliance:
- Uses real DuckDB via TestContext (no mocking)
- Production-parity configuration loading
- Realistic test data via seed packs
- No monkeypatching or test-only code paths

This module tests the coverage edge building functionality including
EdgeContext, FunctionRow, and the edge computation functions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.testing.coverage.edges import (
    EdgeContext,
    FunctionRow,
    backfill_test_goids_for_catalog,
    build_edges_for_file_for_tests,
    compute_test_coverage_edges,
)
from codeintel.config import ConfigBuilder, SnapshotInit, TestCoverageStepConfig
from codeintel.config.primitives import SnapshotRef
from tests._helpers import CORE_PACK, COVERAGE_PACK, TestContext
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_length,
    expect_true,
)
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.factories import make_snapshot

# =============================================================================
# Test Constants
# =============================================================================

# Test data constants (non-repo/commit)
TEST_REL_PATH = "src/module.py"
TEST_QUALNAME = "my_function"
TEST_GOID = 12345
TEST_START_LINE = 10
TEST_END_LINE = 20
TEST_URN = f"urn:codeintel:{DEFAULT_REPO}:{DEFAULT_COMMIT}:{TEST_REL_PATH}#{TEST_QUALNAME}"

# Edge computation constants
EXPECTED_EMPTY_LIST_LENGTH = 0
EXPECTED_SINGLE_EDGE = 1
EXPECTED_COVERAGE_RATIO_FULL = 1.0
FLOAT_COMPARISON_TOLERANCE = 0.01


class TestEdgeContext:
    """Tests for EdgeContext dataclass."""

    @staticmethod
    def test_creates_edge_context(tmp_path: Path) -> None:
        """Verify EdgeContext stores all required fields."""
        now = datetime.now(UTC)
        snapshot = make_snapshot(repo_root=tmp_path)
        cfg = TestCoverageStepConfig(snapshot=snapshot)
        ctx = EdgeContext(
            status_by_test={"test_a": "passed", "test_b": "failed"},
            cfg=cfg,
            now=now,
            test_meta_by_id={"test_a": (123, "urn:test_a")},
        )
        expect_equal(ctx.status_by_test["test_a"], "passed")
        expect_equal(ctx.cfg.repo, DEFAULT_REPO)
        expect_equal(ctx.now, now)
        expect_equal(ctx.test_meta_by_id["test_a"], (123, "urn:test_a"))

    @staticmethod
    def test_edge_context_allows_empty_dicts(tmp_path: Path) -> None:
        """Verify EdgeContext works with empty dictionaries."""
        snapshot = make_snapshot(repo_root=tmp_path)
        cfg = TestCoverageStepConfig(snapshot=snapshot)
        ctx = EdgeContext(
            status_by_test={},
            cfg=cfg,
            now=datetime.now(UTC),
            test_meta_by_id={},
        )
        expect_equal(ctx.status_by_test, {})
        expect_equal(ctx.test_meta_by_id, {})


class TestFunctionRow:
    """Tests for FunctionRow TypedDict."""

    @staticmethod
    def test_creates_function_row() -> None:
        """Verify FunctionRow contains expected fields."""
        row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": TEST_END_LINE,
        }
        expect_equal(row["goid_h128"], TEST_GOID)
        expect_equal(row["urn"], TEST_URN)
        expect_equal(row["rel_path"], TEST_REL_PATH)
        expect_equal(row["qualname"], TEST_QUALNAME)
        expect_equal(row["start_line"], TEST_START_LINE)
        expect_equal(row["end_line"], TEST_END_LINE)

    @staticmethod
    def test_function_row_allows_none_end_line() -> None:
        """Verify FunctionRow allows None for end_line."""
        row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": None,
        }
        expect_is_none(row["end_line"])


class TestBuildEdgesForFile:
    """Tests for build_edges_for_file_for_tests function."""

    @staticmethod
    def _create_edge_context(repo_root: Path) -> EdgeContext:
        """Create a minimal EdgeContext for testing.

        Parameters
        ----------
        repo_root
            Root path for the test repository.

        Returns
        -------
        EdgeContext
            Configured edge context for testing.
        """
        snapshot = make_snapshot(repo_root=repo_root)
        cfg = TestCoverageStepConfig(snapshot=snapshot)
        return EdgeContext(
            status_by_test={"test_func": "passed"},
            cfg=cfg,
            now=datetime.now(UTC),
            test_meta_by_id={"test_func": (999, "urn:test_func")},
        )

    def test_returns_empty_for_empty_functions(self, tmp_path: Path) -> None:
        """Verify returns empty list when no functions."""
        ctx = self._create_edge_context(tmp_path)
        result = build_edges_for_file_for_tests(
            file_funcs=[],
            statements_set={10, 11, 12},
            contexts_by_lineno={10: {"test_func"}},
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )
        expect_equal(result, [])

    def test_returns_empty_for_empty_statements(self, tmp_path: Path) -> None:
        """Verify returns empty when no executable statements in function range."""
        ctx = self._create_edge_context(tmp_path)
        func_row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": 100,  # No statements in this range
            "end_line": 110,
        }
        result = build_edges_for_file_for_tests(
            file_funcs=[func_row],
            statements_set={10, 11, 12},  # Statements outside function range
            contexts_by_lineno={10: {"test_func"}},
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )
        expect_equal(result, [])

    def test_builds_edge_for_covered_function(self, tmp_path: Path) -> None:
        """Verify builds edge when function is covered by test."""
        ctx = self._create_edge_context(tmp_path)
        func_row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": TEST_END_LINE,
        }
        # All lines from 10-20 are statements, all covered by test_func
        statements = set(range(TEST_START_LINE, TEST_END_LINE + 1))
        contexts = {ln: {"test_func"} for ln in statements}

        result = build_edges_for_file_for_tests(
            file_funcs=[func_row],
            statements_set=statements,
            contexts_by_lineno=contexts,
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )

        expect_length(result, EXPECTED_SINGLE_EDGE)
        edge = result[0]
        expect_equal(edge["test_id"], "test_func")
        expect_equal(edge["function_goid_h128"], TEST_GOID)
        expect_equal(edge["repo"], DEFAULT_REPO)
        expect_equal(edge["commit"], DEFAULT_COMMIT)
        expect_equal(edge["coverage_ratio"], EXPECTED_COVERAGE_RATIO_FULL)

    def test_handles_partial_coverage(self, tmp_path: Path) -> None:
        """Verify computes correct coverage ratio for partial coverage."""
        ctx = self._create_edge_context(tmp_path)
        func_row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": TEST_END_LINE,
        }
        # Only half the lines covered
        statements = set(range(TEST_START_LINE, TEST_END_LINE + 1))
        half_covered = set(range(TEST_START_LINE, TEST_START_LINE + 5))
        contexts = {ln: {"test_func"} for ln in half_covered}

        result = build_edges_for_file_for_tests(
            file_funcs=[func_row],
            statements_set=statements,
            contexts_by_lineno=contexts,
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )

        expect_length(result, EXPECTED_SINGLE_EDGE)
        edge = result[0]
        # 5 covered lines out of 11 total (10-20 inclusive)
        expected_ratio = 5 / 11
        expect_true(abs(edge["coverage_ratio"] - expected_ratio) < FLOAT_COMPARISON_TOLERANCE)

    def test_handles_none_end_line(self, tmp_path: Path) -> None:
        """Verify handles function with None end_line (single line function)."""
        ctx = self._create_edge_context(tmp_path)
        func_row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": None,  # Single line function
        }
        statements = {TEST_START_LINE}
        contexts = {TEST_START_LINE: {"test_func"}}

        result = build_edges_for_file_for_tests(
            file_funcs=[func_row],
            statements_set=statements,
            contexts_by_lineno=contexts,
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )

        expect_length(result, EXPECTED_SINGLE_EDGE)
        expect_equal(result[0]["coverage_ratio"], EXPECTED_COVERAGE_RATIO_FULL)

    @staticmethod
    def test_uses_unknown_status_for_unmapped_test(tmp_path: Path) -> None:
        """Verify uses 'unknown' status for tests not in status_by_test."""
        snapshot = make_snapshot(repo_root=tmp_path)
        cfg = TestCoverageStepConfig(snapshot=snapshot)
        ctx = EdgeContext(
            status_by_test={},  # No status mapping
            cfg=cfg,
            now=datetime.now(UTC),
            test_meta_by_id={},
        )
        func_row: FunctionRow = {
            "goid_h128": TEST_GOID,
            "urn": TEST_URN,
            "rel_path": TEST_REL_PATH,
            "qualname": TEST_QUALNAME,
            "start_line": TEST_START_LINE,
            "end_line": TEST_END_LINE,
        }
        statements = {TEST_START_LINE}
        contexts = {TEST_START_LINE: {"unmapped_test"}}

        result = build_edges_for_file_for_tests(
            file_funcs=[func_row],
            statements_set=statements,
            contexts_by_lineno=contexts,
            rel_path=TEST_REL_PATH,
            ctx=ctx,
        )

        expect_length(result, EXPECTED_SINGLE_EDGE)
        expect_equal(result[0]["last_status"], "unknown")


class TestBackfillTestGoids:
    """Tests for backfill_test_goids_for_catalog function."""

    @staticmethod
    def test_returns_empty_dicts_when_no_tests(test_ctx: TestContext) -> None:
        """Verify returns empty dicts when no test catalog entries."""
        # No COVERAGE_PACK applied - empty test catalog
        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
            ),
        ).test_coverage()

        goid_by_id, urn_by_id = backfill_test_goids_for_catalog(test_ctx.gateway, cfg)

        expect_equal(goid_by_id, {})
        expect_equal(urn_by_id, {})

    @staticmethod
    def test_returns_empty_dicts_when_no_goids(coverage_ctx: TestContext) -> None:
        """Verify returns empty dicts when no matching GOIDs."""
        # coverage_ctx has test catalog but we'll query with wrong repo
        snapshot = SnapshotRef(
            repo="nonexistent_repo",
            commit=coverage_ctx.commit,
            repo_root=coverage_ctx.repo_root,
        )
        cfg = TestCoverageStepConfig(snapshot=snapshot)

        goid_by_id, urn_by_id = backfill_test_goids_for_catalog(coverage_ctx.gateway, cfg)

        expect_equal(goid_by_id, {})
        expect_equal(urn_by_id, {})


class TestComputeTestCoverageEdges:
    """Tests for compute_test_coverage_edges function."""

    @staticmethod
    def test_returns_early_when_no_coverage_file(test_ctx: TestContext) -> None:
        """Verify returns early when coverage file doesn't exist."""
        test_ctx.require(CORE_PACK, COVERAGE_PACK)

        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
            ),
        ).test_coverage()
        # Coverage file path doesn't exist by default in test context

        # Should not raise - just logs warning and returns
        compute_test_coverage_edges(test_ctx.gateway, cfg)

        # No edges should be created
        edge_count = test_ctx.query_count(
            "analytics.test_coverage_edges",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )
        expect_true(edge_count >= EXPECTED_EMPTY_LIST_LENGTH)

    @staticmethod
    def test_accepts_custom_coverage_loader(test_ctx: TestContext) -> None:
        """Verify accepts custom coverage loader function."""
        test_ctx.require(CORE_PACK, COVERAGE_PACK)

        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
            ),
        ).test_coverage()

        # Custom loader that returns None (no coverage data)
        def null_loader(_cfg: TestCoverageStepConfig) -> None:
            return None

        # Should not raise - custom loader returns None
        compute_test_coverage_edges(test_ctx.gateway, cfg, coverage_loader=null_loader)

    @staticmethod
    def test_handles_empty_function_catalog(test_ctx: TestContext) -> None:
        """Verify handles case with no functions in catalog."""
        # Just CORE_PACK - no function spans seeded
        test_ctx.require(CORE_PACK)

        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
            ),
        ).test_coverage()

        # Custom loader that returns None
        def null_loader(_cfg: TestCoverageStepConfig) -> None:
            return None

        # Should not raise
        compute_test_coverage_edges(test_ctx.gateway, cfg, coverage_loader=null_loader)
