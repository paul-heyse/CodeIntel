"""Data models plugin.

This plugin extracts structured data models from class definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.data_models import compute_data_models
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import DataModelsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class DataModelsPlugin(TargetPlugin):
    """Extract structured data models from class definitions.

    Extracts from class definitions:
    - Data model schemas
    - Field types and constraints
    - Relationships between models

    Outputs
    -------
    - analytics.data_models: Extracted data models
    """

    plugin_name: ClassVar[str] = "data_models.build"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Extract structured data models from class definitions."

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

        cfg = DataModelsStepConfig(
            snapshot=ctx.snapshot,
        )

        try:
            compute_data_models(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Data models computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["DataModelsPlugin"]
