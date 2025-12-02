"""Coverage functions plugin using the new protocol.

This module provides the coverage functions plugin migrated to the
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
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.config.steps_analytics import CoverageAnalyticsStepConfig


@dataclass
class CoverageFunctionsPlugin:
    """Plugin for aggregating line coverage to function-level metrics.

    Analyzes code coverage data to compute:
    - Function-level coverage percentages
    - Covered/uncovered line counts per function
    - Coverage quality metrics
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="coverage.functions",
            description="Aggregate line coverage to function-level metrics.",
            stage="coverage",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="coverage_functions_cfg",
                    type_ref="CoverageAnalyticsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="coverage_functions",
                    tables=("analytics.coverage_functions",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.coverage_functions", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="coverage.lines", kind="dataset"),),
            depends_on=("goids", "coverage_ingest"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                priority=40,
            ),
            tags=("coverage", "functions"),
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

        if not ctx.has_config(CoverageAnalyticsStepConfig):
            errors.append("CoverageAnalyticsStepConfig is required")

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
            cfg = ctx.get_config(CoverageAnalyticsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        analytics_context = ctx.analytics_context if ctx.has_analytics_context() else None

        try:
            compute_coverage_functions(
                ctx.gateway,
                cfg,
                context=analytics_context,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Coverage functions computation failed: {e}")

        return PluginResult.ok()


__all__ = ["CoverageFunctionsPlugin"]
