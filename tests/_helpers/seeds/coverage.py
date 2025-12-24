"""Coverage seed pack for test coverage data.

This module provides the CoveragePack which seeds test coverage tables:
 test_catalog, test_coverage_edges, and coverage_functions.

The pack depends on CORE_PACK and uses its GOID definitions to create
realistic coverage relationships between tests and production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.fixtures.coverage import CoverageFixtureFactory, CoverageFixtureSpec
from tests._helpers.seeds.core import CORE_PACK

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


@dataclass
class CoveragePack:
    """Seed pack for test coverage data."""

    name: str = "coverage"
    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    passing_ratio: float = 0.75

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one."""
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply coverage seeds to the test context."""
        spec = CoverageFixtureSpec(
            include_catalog=self.include_catalog,
            include_edges=self.include_edges,
            include_functions=self.include_functions,
            include_lines=False,
            passing_ratio=self.passing_ratio,
        )
        CoverageFixtureFactory.seed(ctx, spec)


COVERAGE_PACK = CoveragePack()

__all__ = ["COVERAGE_PACK", "CoveragePack"]
