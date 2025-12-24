"""Coverage lines seed pack for line-level coverage data.

This module provides the CoverageLinesPack which seeds the
analytics.coverage_lines table with realistic line-level coverage data.

The pack depends on CORE_PACK and uses its module paths to create
consistent coverage relationships at the line level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.builders import CoverageLineRow, insert_rows
from tests._helpers.fixtures.repos import MOD_A_PATH, MOD_B_PATH, MOD_C_PATH, MOD_UTIL_PATH
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


FUNCTION_SPANS = {
    GOID_FUNC_A: (MOD_A_PATH, 1, 3),
    GOID_FUNC_B: (MOD_B_PATH, 1, 6),
    GOID_FUNC_C: (MOD_C_PATH, 1, 2),
    GOID_HELPER: (MOD_UTIL_PATH, 1, 2),
}


@dataclass
class CoverageLinesPack:
    """Seed pack for line-level coverage data.

    Seeds analytics.coverage_lines with realistic line-by-line coverage data
    matching the GOIDs and paths from CORE_PACK. This enables testing of
    coverage computation functions that aggregate line data into function
    coverage statistics.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    full_coverage_ratio : float
        Ratio of covered lines for "fully covered" functions (0.0-1.0).
    partial_coverage_ratio : float
        Ratio of covered lines for "partially covered" functions (0.0-1.0).
    include_uncovered_function : bool
        Whether to include lines for an uncovered function.
    """

    name: str = "coverage_lines"
    full_coverage_ratio: float = 1.0
    partial_coverage_ratio: float = 0.6
    include_uncovered_function: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CORE_PACK is required for module paths.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply coverage line seeds to the test context.

        Seeds line-level coverage data for each module in CORE_PACK.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        rows: list[CoverageLineRow] = []

        rows.extend(self._coverage_lines_for_function(ctx, GOID_FUNC_A, self.full_coverage_ratio))
        rows.extend(
            self._coverage_lines_for_function(ctx, GOID_FUNC_B, self.partial_coverage_ratio)
        )
        rows.extend(self._coverage_lines_for_function(ctx, GOID_FUNC_C, self.full_coverage_ratio))
        if self.include_uncovered_function:
            rows.extend(
                self._coverage_lines_for_function(ctx, GOID_HELPER, self.partial_coverage_ratio)
            )

        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _coverage_lines_for_function(
        ctx: TestContext, goid: int, coverage_ratio: float
    ) -> list[CoverageLineRow]:
        """Create coverage line rows for a function's span.

        Parameters
        ----------
        ctx
            Test context for repo/commit.
        goid
            Canonical GOID for the target function (matches CORE_PACK).
        coverage_ratio
            Ratio of lines that should be marked as covered.

        Returns
        -------
        list[CoverageLineRow]
            List of coverage line rows.
        """
        rel_path, start_line, end_line = FUNCTION_SPANS[goid]
        total_lines = max(0, end_line - start_line + 1)
        covered_count = int(total_lines * coverage_ratio)

        rows: list[CoverageLineRow] = []
        for idx, line in enumerate(range(start_line, end_line + 1)):
            is_covered = idx < covered_count
            rows.append(
                CoverageLineRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=rel_path,
                    line=line,
                    is_executable=True,
                    is_covered=is_covered,
                    hits=1 if is_covered else 0,
                    context_count=0,
                )
            )

        return rows


COVERAGE_LINES_PACK = CoverageLinesPack()


__all__ = [
    "COVERAGE_LINES_PACK",
    "FUNCTION_SPANS",
    "CoverageLinesPack",
]
