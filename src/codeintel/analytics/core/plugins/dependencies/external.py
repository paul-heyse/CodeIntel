"""External dependencies plugin using the new protocol."""

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
from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig


@dataclass
class ExternalDepsPlugin:
    """Plugin for identifying external dependency usage.

    Analyzes and builds:
    - External dependency calls per function
    - Aggregated dependency usage patterns
    - Third-party library integration points
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="deps.external",
            description="Identify external dependency usage across functions.",
            stage="other",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="external_deps_cfg",
                    type_ref="ExternalDependenciesStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="external_dependency_calls",
                    tables=("analytics.external_dependency_calls",),
                ),
                PluginOutputSpec(
                    name="external_dependencies",
                    tables=("analytics.external_dependencies",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.external_dependency_calls", kind="dataset"),
                PluginCapability(name="analytics.external_dependencies", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("goids", "config_ingest"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=50,
            ),
            tags=("dependencies", "external", "third-party"),
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
        if not ctx.has_config(ExternalDependenciesStepConfig):
            return ValidationResult.failure(("ExternalDependenciesStepConfig is required",))
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
            cfg = ctx.get_config(ExternalDependenciesStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        analytics_context = ctx.analytics_context if ctx.has_analytics_context() else None
        catalog_provider = analytics_context.catalog if analytics_context else None
        graph_runtime = ctx.graph_runtime if ctx.has_graph_runtime() else None

        try:
            build_external_dependency_calls(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                context=analytics_context,
                runtime=graph_runtime,
            )
            build_external_dependencies(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"External dependencies build failed: {e}")

        return PluginResult.ok()


__all__ = ["ExternalDepsPlugin"]
