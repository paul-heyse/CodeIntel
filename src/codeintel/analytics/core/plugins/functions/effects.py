"""Function effects plugin using the new protocol.

This module provides the function effects plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider
    from codeintel.analytics.resources.graphs import GraphProvider

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
from codeintel.analytics.functions import compute_function_effects
from codeintel.config.steps_analytics import FunctionEffectsStepConfig


@dataclass
class FunctionEffectsPlugin:
    """Plugin for classifying function side effects and purity.

    Analyzes functions to classify:
    - Pure functions vs impure
    - Side effect types (I/O, state mutation, etc.)
    - Effect evidence and reasoning
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="functions.effects",
            description="Classify side effects and purity for functions.",
            stage="function",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="function_effects_cfg",
                    type_ref="FunctionEffectsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="function_effects",
                    tables=("analytics.function_effects",),
                ),
                PluginOutputSpec(
                    name="function_effects_evidence",
                    tables=("analytics.function_effects_evidence",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_effects", kind="dataset"),
                PluginCapability(name="analytics.function_effects_evidence", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("functions.metrics", "callgraph"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=30,
            ),
            tags=("functions", "effects", "purity"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata  # Access self for protocol compliance
        errors: list[str] = []

        if not ctx.has_config(FunctionEffectsStepConfig):
            errors.append("FunctionEffectsStepConfig is required")

        if errors:
            return ValidationResult.failure(tuple(errors))
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
        _ = self.metadata  # Access self for protocol compliance
        try:
            cfg = ctx.get_config(FunctionEffectsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get optional dependencies via resource providers
        analytics_context = None
        catalog_provider = None
        if ctx.has_resource_by_name("AnalyticsContextProvider"):
            provider = cast("AnalyticsContextProvider", ctx.require_by_name("AnalyticsContextProvider"))
            analytics_context = provider.get()
            catalog_provider = analytics_context.catalog if analytics_context else None

        graph_runtime = None
        if ctx.has_resource_by_name("GraphProvider"):
            graph_prov = cast("GraphProvider", ctx.require_by_name("GraphProvider"))
            graph_runtime = graph_prov.runtime

        try:
            compute_function_effects(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                context=analytics_context,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function effects computation failed: {e}")

        return PluginResult.ok()


__all__ = ["FunctionEffectsPlugin"]
