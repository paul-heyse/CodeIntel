"""Subsystems plugin.

This plugin infers subsystems from module coupling and risk signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.analytics.subsystems import build_subsystems
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import SubsystemsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


SUBSYSTEMS_METADATA = CorePluginMetadata(
    name="analytics.subsystems",
    version="3.0.0",
    description="Infer subsystems from module coupling and risk signals.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="subsystem",
    provides=(
        "analytics.subsystems",
        "analytics.subsystem_modules",
        "analytics.subsystem_functions",
    ),
    requires=("graph.call_graph_edges",),
    produces_tables=(
        "analytics.subsystems",
        "analytics.subsystem_modules",
        "analytics.subsystem_functions",
    ),
    consumes_tables=("graph.call_graph_edges",),
)


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

    plugin_name: ClassVar[str] = "subsystems"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Infer subsystems from module coupling and risk signals."
    _core_metadata: ClassVar[CorePluginMetadata] = SUBSYSTEMS_METADATA

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

        cfg = SubsystemsStepConfig(
            snapshot=ctx.snapshot,
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


__all__ = ["SUBSYSTEMS_METADATA", "SubsystemsPlugin"]
