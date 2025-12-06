"""Function effects plugin.

This plugin classifies function side effects and purity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_effects
from codeintel.analytics.functions.function_effects import FunctionEffectsInputs
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionEffectsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class FunctionEffectsPlugin(TargetPlugin):
    """Classify side effects and purity for functions.

    Analyzes functions to classify:
    - Pure functions vs impure
    - Side effect types (I/O, state mutation, etc.)
    - Effect evidence and reasoning

    Outputs
    -------
    - analytics.function_effects: Effect classifications
    - analytics.function_effects_evidence: Effect evidence
    """

    plugin_name: ClassVar[str] = "function_effects"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Classify side effects and purity for functions."

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
        cfg = FunctionEffectsStepConfig(
            snapshot=ctx.snapshot,
        )

        # Get resources
        catalog_provider = ctx.resources.catalog
        graph_runtime = ctx.resources.graph_runtime

        # Get AST data if available
        ast_map = None
        missing_goids = None
        if catalog_provider is not None:
            ast_data = None  # Resources not yet populated via build context
            if ast_data is not None:
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
            return TargetResult.failed(f"Function effects computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FunctionEffectsPlugin"]
