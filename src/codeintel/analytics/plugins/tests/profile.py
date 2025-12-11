"""Test profile plugin.

This plugin builds per-test profiles with coverage and subsystem context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.analytics.testing.profiles.builder import build_test_profile
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import TestProfileStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


TEST_PROFILE_METADATA = CorePluginMetadata(
    name="analytics.test_profile",
    version="3.0.0",
    description="Build per-test profiles with coverage and subsystem context.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="test",
    provides=("analytics.test_profile",),
    requires=("analytics.test_coverage_edges",),
    produces_tables=("analytics.test_profile",),
    consumes_tables=("analytics.test_coverage_edges",),
)


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
    _core_metadata: ClassVar[CorePluginMetadata] = TEST_PROFILE_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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


__all__ = ["TEST_PROFILE_METADATA", "TestProfilePlugin"]
