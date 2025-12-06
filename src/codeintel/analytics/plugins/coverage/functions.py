"""Coverage functions plugin.

This plugin aggregates line coverage to function-level metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.coverage.functions import compute_coverage_functions
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import CoverageAnalyticsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class CoverageFunctionsPlugin(TargetPlugin):
    """Aggregate line coverage to function-level metrics.

    Analyzes code coverage data to compute:
    - Function-level coverage percentages
    - Covered/uncovered line counts per function
    - Coverage quality metrics

    Outputs
    -------
    - analytics.coverage_functions: Function-level coverage metrics
    """

    plugin_name: ClassVar[str] = "coverage_functions"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Aggregate line coverage to function-level metrics."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the coverage functions computation.

        Parameters
        ----------
        ctx
            Execution context with gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self  # Protocol method requires instance

        # Build config from context
        cfg = CoverageAnalyticsStepConfig(
            snapshot=ctx.snapshot,
        )

        try:
            compute_coverage_functions(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Coverage functions computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["CoverageFunctionsPlugin"]
