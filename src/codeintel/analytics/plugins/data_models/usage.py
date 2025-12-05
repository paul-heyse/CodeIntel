"""Data model usage plugin.

This plugin classifies per-function data model read/write usage patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.data_models.usage import compute_data_model_usage
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import DataModelUsageStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class DataModelUsagePlugin(TargetPlugin):
    """Classify per-function data model read/write usage patterns.

    Analyzes per-function:
    - Read/write patterns
    - Model field access
    - Data flow through models

    Outputs
    -------
    - analytics.data_model_usage: Data model usage patterns
    """

    plugin_name: ClassVar[str] = "data_models.usage"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Classify per-function data model read/write usage patterns."
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

        cfg = DataModelUsageStepConfig(
            snapshot=ctx.snapshot,
        )

        # Get resources
        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("Catalog is required for data model usage")

        module_map = None  # Resources not yet populated via build context
        if module_map is None:
            return TargetResult.failed("ModuleMapProvider is required")

        ast_data = None  # Resources not yet populated via build context
        if ast_data is None:
            return TargetResult.failed("AstProvider is required")

        try:
            compute_data_model_usage(
                ctx.gateway,
                cfg,
                module_map=module_map,
                ast_by_goid=ast_data.function_ast_map,
                missing_goids=ast_data.missing_function_goids,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Data model usage computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["DataModelUsagePlugin"]
