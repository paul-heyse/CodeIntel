"""Function contracts plugin using the new protocol.

This module provides the function contracts plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider

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
from codeintel.analytics.functions import compute_function_contracts
from codeintel.config.steps_analytics import FunctionContractsStepConfig


@dataclass
class FunctionContractsPlugin:
    """Plugin for inferring function pre/postconditions and contracts.

    Analyzes functions to infer:
    - Preconditions (required input states)
    - Postconditions (guaranteed output states)
    - Nullability contracts
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="functions.contracts",
            description="Infer pre/postconditions and nullability contracts for functions.",
            stage="function",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="function_contracts_cfg",
                    type_ref="FunctionContractsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="function_contracts",
                    tables=("analytics.function_contracts",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_contracts", kind="dataset"),
            ),
            capabilities_required=(
                PluginCapability(name="analytics.function_metrics", kind="dataset"),
                PluginCapability(name="analytics.docstrings", kind="dataset"),
            ),
            depends_on=("functions.metrics",),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=30,
            ),
            tags=("functions", "contracts", "nullability"),
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

        if not ctx.has_config(FunctionContractsStepConfig):
            errors.append("FunctionContractsStepConfig is required")

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
            cfg = ctx.get_config(FunctionContractsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get required analytics context
        if not ctx.has_resource_by_name("AnalyticsContextProvider"):
            return PluginResult.fail("AnalyticsContextProvider is required")
        analytics_provider = cast(
            "AnalyticsContextProvider", ctx.require_by_name("AnalyticsContextProvider")
        )
        analytics_context = analytics_provider.get()

        try:
            compute_function_contracts(
                ctx.gateway,
                cfg,
                context=analytics_context,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function contracts computation failed: {e}")

        return PluginResult.ok()


__all__ = ["FunctionContractsPlugin"]
