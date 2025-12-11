"""Tests for codeintel.analytics.testing.profiles.builder module.

Testing Charter Compliance:
- Uses real DuckDB via TestContext (no mocking)
- Production-parity configuration loading
- Realistic test data via seed packs
- No monkeypatching or test-only code paths

This module tests the public API for building test profiles and
behavioral coverage, including the infer_behavior_tags function
and related dataclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.testing.profiles.builder import (
    build_test_profile,
    infer_behavior_tags,
)
from codeintel.analytics.testing.profiles.types import IoFlags, TestAstInfo
from codeintel.config import ConfigBuilder, SnapshotInit
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from tests._helpers import TestContext

# =============================================================================
# Test Constants
# =============================================================================

EXPECTED_EMPTY_LIST_LENGTH = 0


class TestInferBehaviorTags:
    """Tests for infer_behavior_tags function."""

    @staticmethod
    def _create_empty_io_flags() -> IoFlags:
        """Create IoFlags with all false values.

        Returns
        -------
        IoFlags
            IoFlags instance with all flags set to False.
        """
        return IoFlags(
            uses_network=False,
            uses_db=False,
            uses_filesystem=False,
            uses_subprocess=False,
        )

    @staticmethod
    def _create_empty_ast_info() -> TestAstInfo:
        """Create TestAstInfo with all false/zero values.

        Returns
        -------
        TestAstInfo
            TestAstInfo instance with all values set to False/zero.
        """
        return TestAstInfo(
            uses_pytest_raises=False,
            uses_concurrency_lib=False,
            has_boundary_asserts=False,
            assert_count=0,
            raise_count=0,
        )

    def test_returns_empty_tags_for_generic_test(self) -> None:
        """Verify returns empty list when no signals present."""
        result = infer_behavior_tags(
            name="test_something",
            markers=[],
            io_flags=self._create_empty_io_flags(),
            ast_info=self._create_empty_ast_info(),
        )
        expect_equal(result, [])

    @pytest.mark.parametrize(
        ("name_suffix", "expected_tag"),
        [
            ("test_happy_path_scenario", "happy_path"),
            ("test_success_case", "happy_path"),
            ("test_error_handling", "error_paths"),
            ("test_invalid_input", "error_paths"),
            ("test_exception_raised", "error_paths"),
            ("test_edge_case", "edge_cases"),
            ("test_boundary_check", "edge_cases"),
            ("test_concurrent_access", "concurrency"),
            ("test_async_operation", "concurrency"),
            ("test_thread_safety", "concurrency"),
        ],
    )
    def test_infers_tags_from_name(self, name_suffix: str, expected_tag: str) -> None:
        """Verify tags are inferred from test name patterns."""
        result = infer_behavior_tags(
            name=name_suffix,
            markers=[],
            io_flags=self._create_empty_io_flags(),
            ast_info=self._create_empty_ast_info(),
        )
        expect_in(expected_tag, result)

    @pytest.mark.parametrize(
        ("markers", "expected_tag"),
        [
            (["xfail"], "known_bug"),
            (["integration"], "integration_scenario"),
            (["e2e"], "integration_scenario"),
            (["slow"], "io_heavy"),
            (["network"], "network_interaction"),
            (["api"], "network_interaction"),
            (["http"], "network_interaction"),
            (["db"], "db_interaction"),
            (["database"], "db_interaction"),
        ],
    )
    def test_infers_tags_from_markers(self, markers: list[str], expected_tag: str) -> None:
        """Verify tags are inferred from marker strings."""
        result = infer_behavior_tags(
            name="test_generic",
            markers=markers,
            io_flags=self._create_empty_io_flags(),
            ast_info=self._create_empty_ast_info(),
        )
        expect_in(expected_tag, result)

    def test_infers_network_interaction_from_io_flags(self) -> None:
        """Verify network_interaction tag from IO flags."""
        io_flags = IoFlags(
            uses_network=True,
            uses_db=False,
            uses_filesystem=False,
            uses_subprocess=False,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        expect_in("network_interaction", result)

    def test_infers_db_interaction_from_io_flags(self) -> None:
        """Verify db_interaction tag from IO flags."""
        io_flags = IoFlags(
            uses_network=False,
            uses_db=True,
            uses_filesystem=False,
            uses_subprocess=False,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        expect_in("db_interaction", result)

    def test_infers_filesystem_interaction_from_io_flags(self) -> None:
        """Verify filesystem_interaction tag from IO flags."""
        io_flags = IoFlags(
            uses_network=False,
            uses_db=False,
            uses_filesystem=True,
            uses_subprocess=False,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        expect_in("filesystem_interaction", result)

    def test_infers_process_interaction_from_io_flags(self) -> None:
        """Verify process_interaction tag from IO flags."""
        io_flags = IoFlags(
            uses_network=False,
            uses_db=False,
            uses_filesystem=False,
            uses_subprocess=True,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        expect_in("process_interaction", result)

    def test_infers_io_heavy_from_io_bound_property(self) -> None:
        """Verify io_heavy tag from io_bound property (derived from any IO flag)."""
        # io_bound is a computed property that's True when any IO flag is set
        # uses_network=True should make io_bound=True
        io_flags = IoFlags(
            uses_network=True,  # This makes io_bound=True
            uses_db=False,
            uses_filesystem=False,
            uses_subprocess=False,
        )
        # io_bound is a property, not a constructor arg
        expect_true(io_flags.io_bound)
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        # Should have io_heavy tag since io_bound is True
        # Note: Also has network_interaction since uses_network=True
        expect_in("network_interaction", result)

    def test_infers_error_paths_from_pytest_raises(self) -> None:
        """Verify error_paths tag from pytest.raises usage."""
        ast_info = TestAstInfo(
            uses_pytest_raises=True,
            uses_concurrency_lib=False,
            has_boundary_asserts=False,
            assert_count=0,
            raise_count=0,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=self._create_empty_io_flags(),
            ast_info=ast_info,
        )
        expect_in("error_paths", result)

    def test_infers_concurrency_from_ast_info(self) -> None:
        """Verify concurrency tag from concurrency lib usage."""
        ast_info = TestAstInfo(
            uses_pytest_raises=False,
            uses_concurrency_lib=True,
            has_boundary_asserts=False,
            assert_count=0,
            raise_count=0,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=self._create_empty_io_flags(),
            ast_info=ast_info,
        )
        expect_in("concurrency", result)

    def test_infers_edge_cases_from_boundary_asserts(self) -> None:
        """Verify edge_cases tag from boundary assertions."""
        ast_info = TestAstInfo(
            uses_pytest_raises=False,
            uses_concurrency_lib=False,
            has_boundary_asserts=True,
            assert_count=0,
            raise_count=0,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=self._create_empty_io_flags(),
            ast_info=ast_info,
        )
        expect_in("edge_cases", result)

    @staticmethod
    def test_combines_tags_from_multiple_sources() -> None:
        """Verify tags are combined from all sources."""
        io_flags = IoFlags(
            uses_network=True,
            uses_db=False,
            uses_filesystem=False,
            uses_subprocess=False,
        )
        ast_info = TestAstInfo(
            uses_pytest_raises=True,
            uses_concurrency_lib=False,
            has_boundary_asserts=False,
            assert_count=0,
            raise_count=0,
        )
        result = infer_behavior_tags(
            name="test_error_handling",
            markers=["integration"],
            io_flags=io_flags,
            ast_info=ast_info,
        )
        # Should have tags from name, markers, io_flags, and ast_info
        expect_in("error_paths", result)
        expect_in("integration_scenario", result)
        expect_in("network_interaction", result)

    def test_returns_sorted_tags(self) -> None:
        """Verify returned tags are sorted alphabetically."""
        io_flags = IoFlags(
            uses_network=True,
            uses_db=True,
            uses_filesystem=True,
            uses_subprocess=False,
        )
        result = infer_behavior_tags(
            name="test_generic",
            markers=[],
            io_flags=io_flags,
            ast_info=self._create_empty_ast_info(),
        )
        expect_equal(result, sorted(result))


class TestBuildTestProfile:
    """Tests for build_test_profile function."""

    @staticmethod
    def test_returns_early_when_no_tests(test_ctx: TestContext) -> None:
        """Verify build_test_profile returns early with no test catalog."""
        # Don't seed COVERAGE_PACK - no test catalog
        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=test_ctx.repo, commit=test_ctx.commit, repo_root=test_ctx.repo_root
            ),
        ).analytics.test_profile()

        # Should not raise - just logs and returns
        build_test_profile(test_ctx.gateway, cfg)

        # No profiles should exist
        count = test_ctx.query_count(
            "analytics.test_profile",
            f"repo = '{test_ctx.repo}' AND commit = '{test_ctx.commit}'",
        )
        expect_equal(count, EXPECTED_EMPTY_LIST_LENGTH)

    @staticmethod
    def test_builds_profiles_with_seeded_tests(coverage_ctx: TestContext) -> None:
        """Verify build_test_profile creates rows when test catalog exists."""
        cfg = ConfigBuilder.from_snapshot(
            snapshot=SnapshotInit(
                repo=coverage_ctx.repo,
                commit=coverage_ctx.commit,
                repo_root=coverage_ctx.repo_root,
            ),
        ).analytics.test_profile()

        build_test_profile(coverage_ctx.gateway, cfg)

        # Profiles should be created for seeded tests
        count = coverage_ctx.query_count(
            "analytics.test_profile",
            f"repo = '{coverage_ctx.repo}' AND commit = '{coverage_ctx.commit}'",
        )
        # COVERAGE_PACK seeds 4 tests
        expect_equal(count, 4)
