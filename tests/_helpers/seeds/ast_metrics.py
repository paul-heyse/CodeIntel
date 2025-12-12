"""AST metrics seed pack for file-level AST analysis data.

This module provides the AstMetricsPack which seeds the core.ast_metrics
table with realistic file-level AST analysis data.

The pack depends on CORE_PACK and uses its module paths to create
consistent AST metrics that can be used for hotspot analysis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import AstMetricsRow, insert_rows
from tests._helpers.seeds.core import (
    CORE_PACK,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


LOW_COMPLEXITY = 3.0
MEDIUM_COMPLEXITY = 7.0
HIGH_COMPLEXITY = 15.0
UTILITY_COMPLEXITY = 2.0


SMALL_NODE_COUNT = 50
MEDIUM_NODE_COUNT = 150
LARGE_NODE_COUNT = 300
UTILITY_NODE_COUNT = 30


@dataclass
class AstMetricsPack:
    """Seed pack for AST metrics data.

    Seeds core.ast_metrics with realistic file-level AST analysis data
    matching the module paths from CORE_PACK. This enables testing of
    hotspot computation and complexity analysis functions.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_high_complexity : bool
        Whether to include a high-complexity file for testing.
    complexity_multiplier : float
        Multiplier for complexity values (for testing different scenarios).
    """

    name: str = "ast_metrics"
    include_high_complexity: bool = True
    complexity_multiplier: float = 1.0

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
        """Apply AST metrics seeds to the test context.

        Seeds AST metrics for each module in CORE_PACK.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        rows: list[AstMetricsRow] = []

        rows.append(
            AstMetricsRow(
                rel_path=MOD_A_PATH,
                node_count=MEDIUM_NODE_COUNT,
                function_count=5,
                class_count=1,
                avg_depth=3.5,
                max_depth=6,
                complexity=MEDIUM_COMPLEXITY * self.complexity_multiplier,
                generated_at=now,
            )
        )

        b_complexity = HIGH_COMPLEXITY if self.include_high_complexity else MEDIUM_COMPLEXITY
        rows.append(
            AstMetricsRow(
                rel_path=MOD_B_PATH,
                node_count=LARGE_NODE_COUNT,
                function_count=12,
                class_count=3,
                avg_depth=4.2,
                max_depth=8,
                complexity=b_complexity * self.complexity_multiplier,
                generated_at=now,
            )
        )

        rows.append(
            AstMetricsRow(
                rel_path=MOD_C_PATH,
                node_count=SMALL_NODE_COUNT,
                function_count=3,
                class_count=0,
                avg_depth=2.0,
                max_depth=4,
                complexity=LOW_COMPLEXITY * self.complexity_multiplier,
                generated_at=now,
            )
        )

        rows.append(
            AstMetricsRow(
                rel_path=MOD_UTIL_PATH,
                node_count=UTILITY_NODE_COUNT,
                function_count=2,
                class_count=0,
                avg_depth=1.5,
                max_depth=3,
                complexity=UTILITY_COMPLEXITY * self.complexity_multiplier,
                generated_at=now,
            )
        )

        insert_rows(ctx.gateway, rows)


AST_METRICS_PACK = AstMetricsPack()


__all__ = [
    "AST_METRICS_PACK",
    "HIGH_COMPLEXITY",
    "LARGE_NODE_COUNT",
    "LOW_COMPLEXITY",
    "MEDIUM_COMPLEXITY",
    "MEDIUM_NODE_COUNT",
    "SMALL_NODE_COUNT",
    "UTILITY_COMPLEXITY",
    "UTILITY_NODE_COUNT",
    "AstMetricsPack",
]
