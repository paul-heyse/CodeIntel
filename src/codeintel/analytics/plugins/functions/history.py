"""Function history plugin.

This plugin aggregates git churn and commit history per function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_history
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionHistoryStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


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
    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_HISTORY_METADATA

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
        _ = self

        max_history_days = ctx.parameters.get("max_history_days", int, default=365)

        cfg = FunctionHistoryStepConfig(
            snapshot=ctx.snapshot,
            max_history_days=max_history_days,
        )

        try:
            compute_function_history(
                ctx.gateway,
                cfg,
                runner=None,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function history computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FUNCTION_HISTORY_METADATA", "FunctionHistoryPlugin"]
