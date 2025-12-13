"""Function effects plugin.

This plugin classifies function side effects and purity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_effects
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


FUNCTION_EFFECTS_METADATA = CorePluginMetadata(
    name="analytics.function_effects",
    version="3.0.0",
    description="Classify side effects and purity for functions.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=(
        "analytics.function_effects",
        "analytics.function_effects_evidence",
    ),
    requires=("graph.call_graph_edges",),
    produces_tables=(
        "analytics.function_effects",
        "analytics.function_effects_evidence",
    ),
    consumes_tables=("graph.call_graph_edges",),
)


class FunctionEffectsPlugin(MetadataPlugin):
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

    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_EFFECTS_METADATA

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

        # Build options from context parameters
        max_call_depth = ctx.parameters.get("max_call_depth", int, default=3)
        require_all_callees_pure = ctx.parameters.get(
            "require_all_callees_pure", bool, default=True
        )
        opts = FunctionEffectsOptions(
            max_call_depth=max_call_depth,
            require_all_callees_pure=require_all_callees_pure,
        )

        catalog_provider = ctx.resources.catalog
        graph_runtime = ctx.resources.graph_runtime

        ast_map = None
        missing_goids = None
        if catalog_provider is not None:
            ast_data = None
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
            compute_function_effects(ctx.gateway, ctx.snapshot, options=opts, inputs=inputs)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function effects computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FUNCTION_EFFECTS_METADATA", "FunctionEffectsPlugin"]
