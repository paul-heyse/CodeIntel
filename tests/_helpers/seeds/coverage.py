"""Coverage seed pack for test coverage data.

This module provides the CoveragePack which seeds test coverage tables:
test_catalog, test_coverage_edges, and coverage_functions.

The pack depends on CORE_PACK and uses its GOID definitions to create
realistic coverage relationships between tests and production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    CoverageFunctionRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_coverage_functions,
    insert_test_catalog,
    insert_test_coverage_edges,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Coverage Data Constants
# =============================================================================

# Test IDs for consistent referencing
TEST_A = "tests/test_mod_a.py::test_func_a"
TEST_B = "tests/test_mod_b.py::test_func_b"
TEST_C = "tests/test_mod_c.py::test_func_c"
TEST_HELPER = "tests/test_util.py::test_helper"


# =============================================================================
# Coverage Pack Implementation
# =============================================================================


@dataclass
class CoveragePack:
    """Seed pack for test coverage data.

    Seeds test catalog, coverage edges, and coverage functions tables
    with consistent test data. Creates realistic coverage relationships
    using GOIDs from CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_catalog : bool
        Whether to seed test catalog.
    include_edges : bool
        Whether to seed coverage edges.
    include_functions : bool
        Whether to seed coverage function summaries.
    passing_ratio : float
        Ratio of passing tests (0.0-1.0).
    """

    name: str = "coverage"
    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    passing_ratio: float = 0.75

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for GOID data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply coverage seeds to the test context.

        Seeds test catalog, coverage edges, and coverage functions.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        if self.include_catalog:
            self._seed_test_catalog(ctx, now)

        if self.include_edges:
            self._seed_coverage_edges(ctx, now)

        if self.include_functions:
            self._seed_coverage_functions(ctx, now)

    def _seed_test_catalog(self, ctx: TestContext, now: datetime) -> None:
        """Seed the test catalog table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        # Determine which tests pass based on ratio
        test_statuses = ["passed", "passed", "passed", "failed"]
        passing_count = int(len(test_statuses) * self.passing_ratio)
        for i in range(passing_count):
            test_statuses[i] = "passed"

        rows = [
            TestCatalogRow(
                test_id=TEST_A,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="tests/test_mod_a.py",
                qualname="test_func_a",
                status=test_statuses[0],
                kind="unit",
                duration_ms=150,
                markers="[]",
                parametrized=False,
                flaky=False,
                created_at=now,
            ),
            TestCatalogRow(
                test_id=TEST_B,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="tests/test_mod_b.py",
                qualname="test_func_b",
                status=test_statuses[1],
                kind="unit",
                duration_ms=200,
                markers='["slow"]',
                parametrized=False,
                flaky=False,
                created_at=now,
            ),
            TestCatalogRow(
                test_id=TEST_C,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="tests/test_mod_c.py",
                qualname="test_func_c",
                status=test_statuses[2],
                kind="integration",
                duration_ms=500,
                markers="[]",
                parametrized=True,
                flaky=False,
                created_at=now,
            ),
            TestCatalogRow(
                test_id=TEST_HELPER,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="tests/test_util.py",
                qualname="test_helper",
                status=test_statuses[3],
                kind="unit",
                duration_ms=50,
                markers="[]",
                parametrized=False,
                flaky=True,
                created_at=now,
            ),
        ]
        insert_test_catalog(ctx.gateway, rows)

    @staticmethod
    def _seed_coverage_edges(ctx: TestContext, now: datetime) -> None:
        """Seed test coverage edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        # Coverage edges: tests cover their respective functions
        edges = [
            # test_func_a covers func_a and helper
            TestCoverageEdgeRow(
                test_id=TEST_A,
                function_goid_h128=GOID_FUNC_A,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                qualname="func_a",
                covered_lines=8,
                executable_lines=10,
                coverage_ratio=0.8,
                last_status="passed",
                created_at=now,
            ),
            TestCoverageEdgeRow(
                test_id=TEST_A,
                function_goid_h128=GOID_HELPER,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                qualname="helper",
                covered_lines=4,
                executable_lines=5,
                coverage_ratio=0.8,
                last_status="passed",
                created_at=now,
            ),
            # test_func_b covers func_b
            TestCoverageEdgeRow(
                test_id=TEST_B,
                function_goid_h128=GOID_FUNC_B,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                qualname="func_b",
                covered_lines=12,
                executable_lines=15,
                coverage_ratio=0.8,
                last_status="passed",
                created_at=now,
            ),
            # test_func_c covers func_c
            TestCoverageEdgeRow(
                test_id=TEST_C,
                function_goid_h128=GOID_FUNC_C,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_C_PATH,
                qualname="func_c",
                covered_lines=6,
                executable_lines=8,
                coverage_ratio=0.75,
                last_status="passed",
                created_at=now,
            ),
        ]
        insert_test_coverage_edges(ctx.gateway, edges)

    @staticmethod
    def _seed_coverage_functions(ctx: TestContext, now: datetime) -> None:
        """Seed coverage function summaries.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            CoverageFunctionRow(
                function_goid_h128=GOID_FUNC_A,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                language="python",
                kind="function",
                qualname="func_a",
                start_line=1,
                end_line=10,
                executable_lines=10,
                covered_lines=8,
                coverage_ratio=0.8,
                tested=True,
                untested_reason=None,
                created_at=now,
            ),
            CoverageFunctionRow(
                function_goid_h128=GOID_FUNC_B,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                language="python",
                kind="function",
                qualname="func_b",
                start_line=1,
                end_line=15,
                executable_lines=15,
                covered_lines=12,
                coverage_ratio=0.8,
                tested=True,
                untested_reason=None,
                created_at=now,
            ),
            CoverageFunctionRow(
                function_goid_h128=GOID_FUNC_C,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_C_PATH,
                language="python",
                kind="function",
                qualname="func_c",
                start_line=1,
                end_line=8,
                executable_lines=8,
                covered_lines=6,
                coverage_ratio=0.75,
                tested=True,
                untested_reason=None,
                created_at=now,
            ),
            CoverageFunctionRow(
                function_goid_h128=GOID_HELPER,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                language="python",
                kind="function",
                qualname="helper",
                start_line=1,
                end_line=5,
                executable_lines=5,
                covered_lines=4,
                coverage_ratio=0.8,
                tested=True,
                untested_reason=None,
                created_at=now,
            ),
        ]
        insert_coverage_functions(ctx.gateway, rows)


# Default instance for common usage
COVERAGE_PACK = CoveragePack()


__all__ = [
    "COVERAGE_PACK",
    "TEST_A",
    "TEST_B",
    "TEST_C",
    "TEST_HELPER",
    "CoveragePack",
]
