"""Semantic roles plugin using the new protocol."""

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
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.config.steps_analytics import SemanticRolesStepConfig


@dataclass
class SemanticRolesPlugin:
    """Plugin for computing semantic roles.

    Classifies functions and calls by:
    - Semantic role (producer, consumer, transformer, etc.)
    - Data flow patterns
    - Behavioral classification
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="semantic.roles",
            description="Compute semantic roles for functions and calls.",
            stage="semantic",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="semantic_roles_cfg",
                    type_ref="SemanticRolesStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="semantic_roles", tables=("analytics.semantic_roles",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.semantic_roles", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("callgraph",),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=50,
            ),
            tags=("semantic", "roles", "classification"),
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
        if not ctx.has_config(SemanticRolesStepConfig):
            return ValidationResult.failure(("SemanticRolesStepConfig is required",))
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
            cfg = ctx.get_config(SemanticRolesStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        analytics_context = ctx.analytics_context if ctx.has_analytics_context() else None
        catalog_provider = analytics_context.catalog if analytics_context else None
        graph_runtime = ctx.graph_runtime if ctx.has_graph_runtime() else None

        try:
            compute_semantic_roles(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                context=analytics_context,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Semantic roles computation failed: {e}")

        return PluginResult.ok()


__all__ = ["SemanticRolesPlugin"]
