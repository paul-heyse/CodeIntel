"""Function history plugin.

This plugin aggregates git churn and commit history per function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_history
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionHistoryStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class FunctionHistoryPlugin(TargetPlugin):
    """Aggregate git churn and commit history per function GOID.

    Analyzes git history to compute:
    - Function churn metrics
    - Commit frequency per function
    - Author contribution patterns

    Outputs
    -------
    - analytics.function_history: History metrics per function
    """

    plugin_name: ClassVar[str] = "function_history"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Aggregate git churn and commit history per function GOID."

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
        max_history_days = ctx.parameters.get("max_history_days", int, default=365)

        cfg = FunctionHistoryStepConfig(
            snapshot=ctx.snapshot,
            max_history_days=max_history_days,
        )

        # Note: ToolRunner type mismatch between build.protocols and ingestion.engine.infrastructure
        # Passing None for now as runner is optional
        try:
            compute_function_history(
                ctx.gateway,
                cfg,
                runner=None,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function history computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FunctionHistoryPlugin"]
