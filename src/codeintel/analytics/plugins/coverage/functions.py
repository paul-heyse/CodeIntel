"""Coverage functions plugin using new base classes.

This module aggregates line coverage data to function-level metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from codeintel.analytics.compute.coverage.functions import compute_coverage_functions
from codeintel.analytics.core.base import ConfiguredTableWriterPlugin
from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    PluginResourceHints,
    PluginStage,
)
from codeintel.config.steps_analytics import CoverageAnalyticsStepConfig


@dataclass
class CoverageFunctionsPlugin(ConfiguredTableWriterPlugin[CoverageAnalyticsStepConfig]):
    """Aggregate line coverage to function-level metrics.

    Analyzes code coverage data to compute:
    - Function-level coverage percentages
    - Covered/uncovered line counts per function
    - Coverage quality metrics
    """

    # Core identification
    plugin_name: ClassVar[str] = "coverage.functions"
    plugin_stage: ClassVar[PluginStage] = "coverage"
    plugin_version: ClassVar[str] = "3.0.0"

    # Configuration binding
    config_type: ClassVar[type[CoverageAnalyticsStepConfig]] = CoverageAnalyticsStepConfig

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = ("analytics.coverage_functions",)

    # Capabilities and dependencies
    provides: ClassVar[tuple[str, ...]] = ("analytics.coverage_functions",)
    requires: ClassVar[tuple[str, ...]] = ("coverage.lines",)
    depends_on: ClassVar[tuple[str, ...]] = ("goids", "coverage_ingest")

    # Categorization
    tags: ClassVar[tuple[str, ...]] = ("coverage", "functions")

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        max_runtime_ms=60_000,
        priority=40,
    )

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute the coverage functions computation.

        Parameters
        ----------
        ctx
            Execution context with gateway and config.

        Returns
        -------
        Mapping[str, int] | None
            None to trigger auto row count computation.
        """
        compute_coverage_functions(ctx.gateway, self.config)
        return None  # Let base class compute row counts


__all__ = ["CoverageFunctionsPlugin"]
