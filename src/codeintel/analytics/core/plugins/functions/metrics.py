"""Function metrics plugin - thin orchestration layer.

This plugin delegates computation to the pure compute layer and
persistence to the adapters layer. The plugin itself is minimal:
- Declares metadata via class attributes
- Orchestrates execution in compute()
- Base classes handle validation, contracts, and error handling

Architecture
------------
- Compute: `analytics.compute.functions` (pure, no I/O)
- Adapters: `analytics.adapters.functions` (database I/O)
- Plugin: This module (thin orchestration)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.asts import AstProvider

from codeintel.analytics.core.base import ConfiguredTableWriterPlugin
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import PluginResourceHints, PluginStage
from codeintel.analytics.core.traits import WithContractValidation
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig


@dataclass
class FunctionMetricsPlugin(
    ConfiguredTableWriterPlugin[FunctionAnalyticsStepConfig],
    WithContractValidation,
):
    """Compute function metrics, complexity, and type annotations.

    Orchestrates function analysis by delegating to:
    - Compute layer for pure metric computation
    - Adapters for database persistence

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    # Identification
    plugin_name: ClassVar[str] = "functions.metrics"
    plugin_stage: ClassVar[PluginStage] = "function"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute function complexity and type coverage metrics"

    # Configuration
    config_type: ClassVar[type[FunctionAnalyticsStepConfig]] = FunctionAnalyticsStepConfig

    # Output tables (contracts auto-generated from these)
    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.function_metrics",
        "analytics.function_types",
    )

    # Capabilities
    provides: ClassVar[tuple[str, ...]] = output_tables
    requires: ClassVar[tuple[str, ...]] = ("core.goids",)
    tags: ClassVar[tuple[str, ...]] = ("functions", "metrics", "types")

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        max_runtime_ms=60_000,
        priority=10,
    )

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute function metrics computation.

        Parameters
        ----------
        ctx
            Execution context providing gateway, config, and resources.

        Returns
        -------
        Mapping[str, int] | None
            Row counts per output table.
        """
        # Get AST data from AstProvider if available
        function_ast_map = None
        missing_function_goids: set[int] = set()

        if ctx.has_resource_by_name("AstProvider"):
            ast_provider = cast("AstProvider", ctx.require_by_name("AstProvider"))
            ast_data = ast_provider.get()
            function_ast_map = ast_data.function_ast_map
            missing_function_goids = ast_data.missing_function_goids

        opts = FunctionAnalyticsOptions(
            function_ast_map=function_ast_map,
            missing_function_goids=missing_function_goids,
        )

        result = compute_function_metrics_and_types(ctx.gateway, self.config, options=opts)

        # Map legacy counter names to table names
        return {
            "analytics.function_metrics": result.get("metrics_rows", 0),
            "analytics.function_types": result.get("types_rows", 0),
        }


__all__ = ["FunctionMetricsPlugin"]
