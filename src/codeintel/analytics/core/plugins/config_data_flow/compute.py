"""Config data flow plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
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
from codeintel.analytics.graphs import compute_config_data_flow
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig


@dataclass
class ConfigDataFlowPlugin:
    """Plugin for tracking configuration key data flow.

    Tracks configuration usage:
    - Config key reads at function level
    - Config key propagation through calls
    - Function-level config dependencies
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="config.data_flow",
            description="Track configuration key usage and data flow at the function level.",
            stage="config",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="config_data_flow_cfg",
                    type_ref="ConfigDataFlowStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="config_data_flow", tables=("analytics.config_data_flow",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.config_data_flow", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.config_keys", kind="dataset"),),
            depends_on=("config_ingest", "callgraph", "functions.metrics", "entrypoints.build"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=40,
            ),
            tags=("config", "data_flow", "tracking"),
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
        if not ctx.has_config(ConfigDataFlowStepConfig):
            return ValidationResult.failure(("ConfigDataFlowStepConfig is required",))
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
            cfg = ctx.get_config(ConfigDataFlowStepConfig)
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

        try:
            compute_config_data_flow(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Config data flow computation failed: {e}")

        return PluginResult.ok()


__all__ = ["ConfigDataFlowPlugin"]
