"""Data models plugin.

This plugin extracts structured data models from class definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.data_models import compute_data_models
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


DATA_MODELS_METADATA = CorePluginMetadata(
    name="analytics.data_models",
    version="3.0.0",
    description="Extract structured data models from class definitions.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="data_model",
    provides=(
        "analytics.data_models",
        "analytics.data_model_fields",
        "analytics.data_model_relationships",
    ),
    requires=("core.modules", "core.goids", "core.ast_metrics"),
    produces_tables=(
        "analytics.data_models",
        "analytics.data_model_fields",
        "analytics.data_model_relationships",
    ),
    consumes_tables=("core.modules", "core.goids", "core.ast_metrics"),
)


class DataModelsPlugin(MetadataPlugin):
    """Extract structured data models from class definitions.

    Extracts from class definitions:
    - Data model schemas
    - Field types and constraints
    - Relationships between models

    Outputs
    -------
    - analytics.data_models: Extracted data models
    """

    _core_metadata: ClassVar[CorePluginMetadata] = DATA_MODELS_METADATA

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

        try:
            compute_data_models(ctx.gateway, ctx.snapshot)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Data models computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["DATA_MODELS_METADATA", "DataModelsPlugin"]
