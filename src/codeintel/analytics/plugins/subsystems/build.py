"""Subsystems plugin.

This plugin infers subsystems from module coupling and risk signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.subsystems import build_subsystems
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import SubsystemsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class SubsystemsPlugin(TargetPlugin):
    """Infer subsystems from module coupling and risk signals.

    Detects and builds:
    - Subsystem boundaries from coupling analysis
    - Module to subsystem mappings
    - Function to subsystem associations

    Outputs
    -------
    - analytics.subsystems: Subsystem definitions
    - analytics.subsystem_modules: Module to subsystem mappings
    - analytics.subsystem_functions: Function to subsystem associations
    """

    plugin_name: ClassVar[str] = "subsystems.build"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Infer subsystems from module coupling and risk signals."
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
        cfg = SubsystemsStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        graph_runtime = ctx.resources.graph_runtime

        try:
            build_subsystems(
                ctx.gateway,
                cfg,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Subsystems build failed: {e}")

        return TargetResult.succeeded()


__all__ = ["SubsystemsPlugin"]
