"""Hotspots plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginCapability,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.config.steps_analytics import HotspotsStepConfig


@dataclass
class HotspotsPlugin:
    """Plugin for computing file-level hotspots.

    Identifies high-risk code areas based on:
    - AST complexity metrics
    - Git churn patterns
    - Change frequency
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="hotspots.build",
            description="Compute file-level hotspots from AST metrics and churn.",
            stage="hotspots",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="hotspots_cfg",
                    type_ref="HotspotsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="hotspots",
                    tables=("analytics.hotspots",),
                ),
            ),
            capabilities_provided=(PluginCapability(name="analytics.hotspots", kind="dataset"),),
            capabilities_required=(PluginCapability(name="core.ast_metrics", kind="dataset"),),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                priority=50,
            ),
            tags=("hotspots", "risk", "churn"),
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
        if not ctx.has_config(HotspotsStepConfig):
            return ValidationResult.failure(("HotspotsStepConfig is required",))
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
            cfg = ctx.get_config(HotspotsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        try:
            build_hotspots(ctx.gateway, cfg, runner=ctx.extra.get("tool_runner"))
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Hotspots build failed: {e}")

        return PluginResult.ok()


__all__ = ["HotspotsPlugin"]
