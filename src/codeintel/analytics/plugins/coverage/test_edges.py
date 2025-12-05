"""Coverage test edges plugin.

This plugin builds test-to-function coverage edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.testing import compute_test_coverage_edges
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import TestCoverageStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class CoverageTestEdgesPlugin(TargetPlugin):
    """Build test-to-function coverage edges from coverage contexts.

    Analyzes coverage context data to build:
    - Test to function mapping
    - Coverage relationship edges
    - Test impact analysis foundation

    Outputs
    -------
    - coverage.test_edges: Test-to-function coverage edges
    """

    plugin_name: ClassVar[str] = "coverage.test_edges"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Build test-to-function coverage edges from coverage contexts."
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

        # Build config from context
        cfg = TestCoverageStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        catalog_provider = ctx.resources.catalog

        try:
            compute_test_coverage_edges(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Coverage test edges computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["CoverageTestEdgesPlugin"]
