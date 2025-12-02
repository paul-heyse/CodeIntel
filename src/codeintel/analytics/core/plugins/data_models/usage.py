"""Data model usage plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.catalog import CatalogProvider

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
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.config.steps_analytics import DataModelUsageStepConfig


@dataclass
class DataModelUsagePlugin:
    """Plugin for classifying data model usage patterns.

    Analyzes per-function:
    - Read/write patterns
    - Model field access
    - Data flow through models
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="data_models.usage",
            description="Classify per-function data model read/write usage patterns.",
            stage="data_model_usage",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="data_model_usage_cfg",
                    type_ref="DataModelUsageStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="data_model_usage", tables=("analytics.data_model_usage",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.data_model_usage", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="analytics.data_models", kind="dataset"),),
            depends_on=("data_models.build", "callgraph", "cfg", "functions.metrics"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=45,
            ),
            tags=("data_models", "usage", "patterns"),
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
        if not ctx.has_config(DataModelUsageStepConfig):
            return ValidationResult.failure(("DataModelUsageStepConfig is required",))
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
            cfg = ctx.get_config(DataModelUsageStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get catalog from CatalogProvider
        catalog_provider = None
        if ctx.has_resource_by_name("CatalogProvider"):
            cat_prov = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog_provider = cat_prov.get()

        try:
            compute_data_model_usage(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Data model usage computation failed: {e}")

        return PluginResult.ok()


__all__ = ["DataModelUsagePlugin"]
