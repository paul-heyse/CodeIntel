"""Function metrics plugin.

This plugin computes function complexity and type coverage metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class FunctionMetricsPlugin(TargetPlugin):
    """Compute function complexity and type coverage metrics.

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    plugin_name: ClassVar[str] = "function_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute function complexity and type coverage metrics."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute function metrics computation.

        Parameters
        ----------
        ctx
            Execution context providing gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self  # Protocol method requires instance

        # Get AST data from catalog if available
        function_ast_map = None
        missing_function_goids: set[int] = set()

        # Note: Direct resource access not yet implemented in build context
        # The FunctionCatalogProvider doesn't have get_resource()
        # This will be populated when the build executor provides resources

        opts = FunctionAnalyticsOptions(
            function_ast_map=function_ast_map,
            missing_function_goids=missing_function_goids,
        )

        # Build config from parameters
        cfg = FunctionAnalyticsStepConfig(
            snapshot=ctx.snapshot,
        )

        result = compute_function_metrics_and_types(ctx.gateway, cfg, options=opts)

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result.get("metrics_rows", 0),
                "analytics.function_types": result.get("types_rows", 0),
            }
        )


__all__ = ["FunctionMetricsPlugin"]
