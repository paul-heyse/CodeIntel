"""Function history plugin.

This plugin aggregates git churn and commit history per function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_history
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


FUNCTION_HISTORY_METADATA = CorePluginMetadata(
    name="analytics.function_history",
    version="3.0.0",
    description="Aggregate git churn and commit history per function GOID.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function_history",
    provides=("analytics.function_history",),
    requires=("core.goids",),
    produces_tables=("analytics.function_history",),
    consumes_tables=("core.goids",),
)


class FunctionHistoryPlugin(MetadataPlugin):
    """Aggregate git churn and commit history per function GOID.

    Analyzes git history to compute:
    - Function churn metrics
    - Commit frequency per function
    - Author contribution patterns

    Outputs
    -------
    - analytics.function_history: History metrics per function
    """

    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_HISTORY_METADATA

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
        _ = self

        try:
            compute_function_history(
                ctx.gateway,
                ctx.snapshot,
                runner=None,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function history computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FUNCTION_HISTORY_METADATA", "FunctionHistoryPlugin"]
