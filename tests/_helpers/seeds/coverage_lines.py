"""Coverage lines seed pack for line-level coverage data.

This module provides the CoverageLinesPack which seeds the
analytics.coverage_lines table with realistic line-level coverage data.

The pack depends on CORE_PACK and uses its module paths to create
consistent coverage relationships at the line level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.fixtures.coverage import CoverageFixtureFactory, CoverageFixtureSpec
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
    GOID_FUNC_A: (MOD_A_PATH, 1, 10),
    GOID_FUNC_B: (MOD_B_PATH, 1, 15),
    GOID_FUNC_C: (MOD_C_PATH, 1, 8),
    GOID_HELPER: (MOD_UTIL_PATH, 1, 5),
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
    full_coverage_ratio: float = 0.8
    partial_coverage_ratio: float = 0.75
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
        line_coverage: dict[int, float] = {
            GOID_FUNC_A: self.full_coverage_ratio,
            GOID_FUNC_B: self.full_coverage_ratio,
            GOID_FUNC_C: self.partial_coverage_ratio,
        }
        if self.include_uncovered_function:
            line_coverage[GOID_HELPER] = self.partial_coverage_ratio

        spec = CoverageFixtureSpec(
            include_catalog=False,
            include_edges=False,
            include_functions=False,
            include_lines=True,
            line_spans=FUNCTION_SPANS,
            line_coverage=line_coverage,
        )
        CoverageFixtureFactory.seed(ctx, spec)


COVERAGE_LINES_PACK = CoverageLinesPack()


__all__ = [
    "COVERAGE_LINES_PACK",
    "FUNCTION_SPANS",
    "CoverageLinesPack",
]
