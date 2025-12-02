"""Function metrics plugin using the new protocol.

This module provides the function metrics plugin migrated to the
new unified plugin protocol, demonstrating the migration pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.core.contracts import OutputContractSpec
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
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig


@dataclass
class FunctionMetricsPlugin:
    """Plugin for computing function metrics and type annotations.

    This plugin analyzes functions in the codebase to compute:
    - Code complexity metrics (cyclomatic, cognitive)
    - Type annotation coverage
    - Function signatures and parameters
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="functions.metrics",
            description="Compute function metrics, complexity, and type annotations.",
            stage="function",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="function_cfg",
                    type_ref="FunctionAnalyticsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="function_metrics",
                    tables=("analytics.function_metrics",),
                    min_rows=1,
                    required_columns=("repo", "commit", "goid", "complexity", "lines"),
                ),
                PluginOutputSpec(
                    name="function_types",
                    tables=("analytics.function_types",),
                    min_rows=1,
                    required_columns=("repo", "commit", "goid"),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_metrics", kind="dataset"),
                PluginCapability(name="analytics.function_types", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                requires_gpu=False,
                priority=10,
            ),
            tags=("functions", "metrics", "types"),
        )

    @property
    def requires_catalog(self) -> bool:
        """Return whether catalog is required."""
        return True

    @property
    def output_contracts(self) -> tuple[OutputContractSpec, ...]:
        """Return output contracts for validation."""
        return (
            OutputContractSpec(
                table="analytics.function_metrics",
                min_rows=1,
                required_columns=("repo", "commit", "goid", "complexity", "lines"),
                description="Function complexity and size metrics",
            ),
            OutputContractSpec(
                table="analytics.function_types",
                min_rows=1,
                required_columns=("repo", "commit", "goid"),
                description="Function type annotation data",
            ),
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

        # Check for required config
        if not ctx.has_config(FunctionAnalyticsStepConfig):
            errors.append("FunctionAnalyticsStepConfig is required")

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
            cfg = ctx.get_config(FunctionAnalyticsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Build options from context
        analytics_context = ctx.analytics_context if ctx.has_analytics_context() else None
        opts = FunctionAnalyticsOptions(context=analytics_context)

        try:
            counters = compute_function_metrics_and_types(
                ctx.gateway,
                cfg,
                options=opts,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function metrics computation failed: {e}")

        # Convert counters to row counts
        row_counts: dict[str, int] = {}
        if isinstance(counters, dict):
            if "metrics_written" in counters:
                row_counts["analytics.function_metrics"] = counters["metrics_written"]
            if "types_written" in counters:
                row_counts["analytics.function_types"] = counters["types_written"]

        return PluginResult.ok(
            row_counts=row_counts,
            meta=counters if isinstance(counters, dict) else {},
        )


__all__ = [
    "FunctionMetricsPlugin",
]
