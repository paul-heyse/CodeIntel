"""Function history plugin using the new protocol.

This module provides the function history plugin migrated to the
new unified plugin protocol.
"""

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
from codeintel.analytics.functions import compute_function_history
from codeintel.config.steps_analytics import FunctionHistoryStepConfig


@dataclass
class FunctionHistoryPlugin:
    """Plugin for aggregating git churn and commit history per function.

    Analyzes git history to compute:
    - Function churn metrics
    - Commit frequency per function
    - Author contribution patterns
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="functions.history",
            description="Aggregate git churn and commit history per function GOID.",
            stage="function_history",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="function_history_cfg",
                    type_ref="FunctionHistoryStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="function_history",
                    tables=("analytics.function_history",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_history", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("functions.metrics", "hotspots.build"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                priority=40,
            ),
            tags=("functions", "history", "git"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata  # Access self for protocol compliance
        errors: list[str] = []

        if not ctx.has_config(FunctionHistoryStepConfig):
            errors.append("FunctionHistoryStepConfig is required")

        if errors:
            return ValidationResult.failure(tuple(errors))
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
        _ = self.metadata  # Access self for protocol compliance
        try:
            cfg = ctx.get_config(FunctionHistoryStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get optional tool runner from context extras
        tool_runner = ctx.extra.get("tool_runner")

        try:
            # Function history no longer requires AnalyticsContext
            # The domain function works directly with database queries
            compute_function_history(
                ctx.gateway,
                cfg,
                runner=tool_runner,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function history computation failed: {e}")

        return PluginResult.ok()


__all__ = ["FunctionHistoryPlugin"]
