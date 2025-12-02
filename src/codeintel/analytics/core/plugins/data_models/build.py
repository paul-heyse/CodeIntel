"""Data models plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass

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
from codeintel.analytics.data_models import compute_data_models
from codeintel.config.steps_analytics import DataModelsStepConfig


@dataclass
class DataModelsPlugin:
    """Plugin for extracting structured data models.

    Extracts from class definitions:
    - Data model schemas
    - Field types and constraints
    - Relationships between models
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="data_models.build",
            description="Extract structured data models from class definitions.",
            stage="data_model",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="data_models_cfg",
                    type_ref="DataModelsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(PluginOutputSpec(name="data_models", tables=("analytics.data_models",)),),
            capabilities_provided=(PluginCapability(name="analytics.data_models", kind="dataset"),),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("ast_extract", "goids", "docstrings_ingest"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                priority=40,
            ),
            tags=("data_models", "schema", "extraction"),
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
        if not ctx.has_config(DataModelsStepConfig):
            return ValidationResult.failure(("DataModelsStepConfig is required",))
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
            cfg = ctx.get_config(DataModelsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        try:
            compute_data_models(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Data models computation failed: {e}")

        return PluginResult.ok()


__all__ = ["DataModelsPlugin"]
