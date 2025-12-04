"""Subsystems plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.graphs import GraphProvider

from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.subsystems import build_subsystems
from codeintel.config.steps_analytics import SubsystemsStepConfig


@dataclass
class SubsystemsPlugin:
    """Plugin for inferring subsystems from module coupling.

    Detects and builds:
    - Subsystem boundaries from coupling analysis
    - Module to subsystem mappings
    - Function to subsystem associations
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="subsystems.build",
            description="Infer subsystems from module coupling and risk signals.",
            kind="analytics",
            stage="subsystem",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="subsystems_cfg",
                    type_ref="SubsystemsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="subsystems", tables=("analytics.subsystems",)),
                PluginOutputSpec(name="subsystem_modules", tables=("analytics.subsystem_modules",)),
                PluginOutputSpec(
                    name="subsystem_functions", tables=("analytics.subsystem_functions",)
                ),
            ),
            provides=(
                "analytics.subsystems",
                "analytics.subsystem_modules",
                "analytics.subsystem_functions",
            ),
            requires=("core.modules",),
            depends_on=("import_graph", "symbol_uses", "risk_factors.build"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                priority=60,
            ),
            tags=("subsystems", "architecture", "modules"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate required inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata
        if not ctx.has_config(SubsystemsStepConfig):
            return ValidationResult.failure(("SubsystemsStepConfig is required",))
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Execution result.
        """
        _ = self.metadata
        try:
            cfg = ctx.get_config(SubsystemsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get graph runtime from GraphProvider (optional)
        graph_runtime = None
        if ctx.has_resource_by_name("GraphProvider"):
            graph_prov = cast("GraphProvider", ctx.require_by_name("GraphProvider"))
            graph_runtime = graph_prov.runtime

        try:
            build_subsystems(
                ctx.gateway,
                cfg,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Subsystems build failed: {e}")

        return PluginResult.ok()


__all__ = ["SubsystemsPlugin"]
