"""Test profile plugin.

This plugin builds per-test profiles with coverage and subsystem context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.testing.profiles.builder import build_test_profile
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import TestProfileStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class TestProfilePlugin(TargetPlugin):
    """Build per-test profiles with coverage and subsystem context.

    Creates comprehensive test profiles including:
    - Coverage context for each test
    - Subsystem associations
    - Test metadata aggregation

    Outputs
    -------
    - analytics.test_profile: Per-test profiles
    """

    plugin_name: ClassVar[str] = "test_profile"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Build per-test profiles with coverage and subsystem context."
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self  # Protocol method requires instance

        cfg = TestProfileStepConfig(
            snapshot=ctx.snapshot,
        )

        try:
            build_test_profile(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Test profile build failed: {e}")

        # Count rows written
        row = ctx.gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.test_profile
            WHERE repo = ? AND commit = ?
            """,
            [ctx.repo, ctx.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return TargetResult.succeeded(
            row_counts={"analytics.test_profile": row_count},
        )


__all__ = ["TestProfilePlugin"]
