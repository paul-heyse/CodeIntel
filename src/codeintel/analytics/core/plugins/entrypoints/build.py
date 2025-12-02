"""Entrypoints plugin using the new protocol."""

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
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.config.steps_analytics import EntryPointsStepConfig


@dataclass
class EntrypointsPlugin:
    """Plugin for detecting application entrypoints.

    Identifies and maps:
    - HTTP endpoints (Flask, FastAPI, etc.)
    - CLI commands
    - Background job handlers
    - Tests covering each entrypoint
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="entrypoints.build",
            description="Detect HTTP/CLI/job entrypoints and map them to handlers and tests.",
            stage="entrypoints",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="entrypoints_cfg",
                    type_ref="EntryPointsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="entrypoints", tables=("analytics.entrypoints",)),
                PluginOutputSpec(name="entrypoint_tests", tables=("analytics.entrypoint_tests",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.entrypoints", kind="dataset"),
                PluginCapability(name="analytics.entrypoint_tests", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("subsystems.build", "coverage.functions", "coverage.test_edges", "goids"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=50,
            ),
            tags=("entrypoints", "http", "cli"),
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
        if not ctx.has_config(EntryPointsStepConfig):
            return ValidationResult.failure(("EntryPointsStepConfig is required",))
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
            cfg = ctx.get_config(EntryPointsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        analytics_context = ctx.analytics_context if ctx.has_analytics_context() else None
        catalog_provider = analytics_context.catalog if analytics_context else None
        graph_runtime = ctx.graph_runtime if ctx.has_graph_runtime() else None

        try:
            build_entrypoints(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                context=analytics_context,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Entrypoints build failed: {e}")

        return PluginResult.ok()


__all__ = ["EntrypointsPlugin"]
