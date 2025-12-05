"""Config data flow plugin.

This plugin tracks configuration key usage and data flow at the function level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs import compute_config_data_flow
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class ConfigDataFlowPlugin(TargetPlugin):
    """Track configuration key usage and data flow at the function level.

    Tracks configuration usage:
    - Config key reads at function level
    - Config key propagation through calls
    - Function-level config dependencies

    Outputs
    -------
    - analytics.config_data_flow: Config data flow tracking
    """

    plugin_name: ClassVar[str] = "config.data_flow"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Track configuration key usage and data flow at the function level."
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

        cfg = ConfigDataFlowStepConfig(
            snapshot=ctx.snapshot,
        )

        catalog_provider = ctx.resources.catalog
        graph_runtime = ctx.resources.graph_runtime

        try:
            compute_config_data_flow(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Config data flow computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["ConfigDataFlowPlugin"]
