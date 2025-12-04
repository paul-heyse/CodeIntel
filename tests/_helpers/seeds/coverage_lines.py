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
from tests._helpers.seeds.core import (
    CORE_PACK,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Coverage Lines Constants
# =============================================================================

# Line ranges for each module (matching CORE_PACK GOIDs)
MOD_A_START = 1
MOD_A_END = 10
MOD_B_START = 1
MOD_B_END = 15
MOD_C_START = 1
MOD_C_END = 8
MOD_UTIL_START = 1
MOD_UTIL_END = 5


# =============================================================================
# Coverage Lines Pack Implementation
# =============================================================================


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

        # Module A: fully covered
        rows.extend(
            self._make_coverage_lines(
                ctx=ctx,
                rel_path=MOD_A_PATH,
                start_line=MOD_A_START,
                end_line=MOD_A_END,
                coverage_ratio=self.full_coverage_ratio,
            )
        )

        # Module B: partially covered (some lines not covered)
        rows.extend(
            self._make_coverage_lines(
                ctx=ctx,
                rel_path=MOD_B_PATH,
                start_line=MOD_B_START,
                end_line=MOD_B_END,
                coverage_ratio=self.partial_coverage_ratio,
            )
        )

        # Module C: fully covered
        rows.extend(
            self._make_coverage_lines(
                ctx=ctx,
                rel_path=MOD_C_PATH,
                start_line=MOD_C_START,
                end_line=MOD_C_END,
                coverage_ratio=self.full_coverage_ratio,
            )
        )

        # Utility module: partially covered
        rows.extend(
            self._make_coverage_lines(
                ctx=ctx,
                rel_path=MOD_UTIL_PATH,
                start_line=MOD_UTIL_START,
                end_line=MOD_UTIL_END,
                coverage_ratio=self.partial_coverage_ratio,
            )
        )

        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _make_coverage_lines(
        ctx: TestContext,
        rel_path: str,
        start_line: int,
        end_line: int,
        coverage_ratio: float,
    ) -> list[CoverageLineRow]:
        """Create coverage line rows for a function's line range.

        Parameters
        ----------
        ctx
            Test context for repo/commit.
        rel_path
            Relative path to the source file.
        start_line
            First line number of the function.
        end_line
            Last line number of the function.
        coverage_ratio
            Ratio of lines that should be marked as covered.

        Returns
        -------
        list[CoverageLineRow]
            List of coverage line rows.
        """
        rows: list[CoverageLineRow] = []
        total_lines = end_line - start_line + 1
        covered_count = int(total_lines * coverage_ratio)

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


# Default instance for common usage
COVERAGE_LINES_PACK = CoverageLinesPack()


__all__ = [
    "COVERAGE_LINES_PACK",
    "MOD_A_END",
    "MOD_A_START",
    "MOD_B_END",
    "MOD_B_START",
    "MOD_C_END",
    "MOD_C_START",
    "MOD_UTIL_END",
    "MOD_UTIL_START",
    "CoverageLinesPack",
]
