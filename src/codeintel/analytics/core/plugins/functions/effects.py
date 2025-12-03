"""Function effects plugin using the new protocol.

This module provides the function effects plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.asts import AstProvider
    from codeintel.analytics.resources.catalog import CatalogProvider
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
from codeintel.analytics.functions.function_effects import FunctionEffectsInputs
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
            version="3.0.0",
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

        # Get catalog from CatalogProvider
        catalog_provider = None
        if ctx.has_resource_by_name("CatalogProvider"):
            cat_prov = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog_provider = cat_prov.get()

        # Get graph runtime from GraphProvider
        graph_runtime = None
        if ctx.has_resource_by_name("GraphProvider"):
            graph_prov = cast("GraphProvider", ctx.require_by_name("GraphProvider"))
            graph_runtime = graph_prov.runtime

        # Get AST data from AstProvider (AstResourceData)
        ast_map = None
        missing_goids = None
        if ctx.has_resource_by_name("AstProvider"):
            ast_prov = cast("AstProvider", ctx.require_by_name("AstProvider"))
            ast_data = ast_prov.get()
            ast_map = ast_data.function_ast_map
            missing_goids = ast_data.missing_function_goids

        try:
            inputs = FunctionEffectsInputs(
                catalog_provider=catalog_provider,
                runtime=graph_runtime,
                ast_map=ast_map,
                missing_goids=missing_goids,
            )
            compute_function_effects(ctx.gateway, cfg, inputs=inputs)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function effects computation failed: {e}")

        return PluginResult.ok()


__all__ = ["FunctionEffectsPlugin"]
