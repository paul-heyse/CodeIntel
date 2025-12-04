"""Hotspots plugin using new base classes.

This module computes file-level hotspots based on AST complexity
metrics and Git churn patterns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from codeintel.analytics.compute.hotspots.metrics import build_hotspots
from codeintel.analytics.core.base import ConfiguredTableWriterPlugin
from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    PluginResourceHints,
    PluginStage,
)
from codeintel.config.steps_analytics import HotspotsStepConfig


@dataclass
class HotspotsPlugin(ConfiguredTableWriterPlugin[HotspotsStepConfig]):
    """Compute file-level hotspots from AST metrics and churn.

    Identifies high-risk code areas based on:
    - AST complexity metrics
    - Git churn patterns
    - Change frequency
    """

    # Core identification
    plugin_name: ClassVar[str] = "hotspots.build"
    plugin_stage: ClassVar[PluginStage] = "hotspots"
    plugin_version: ClassVar[str] = "2.0.0"

    # Configuration binding
    config_type: ClassVar[type[HotspotsStepConfig]] = HotspotsStepConfig

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = ("analytics.hotspots",)

    # Capabilities
    provides: ClassVar[tuple[str, ...]] = ("analytics.hotspots",)
    requires: ClassVar[tuple[str, ...]] = ("core.ast_metrics",)

    # Categorization
    tags: ClassVar[tuple[str, ...]] = ("hotspots", "risk", "churn")

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        max_runtime_ms=60_000,
        priority=50,
    )

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute the hotspots computation.

        Parameters
        ----------
        ctx
            Execution context with gateway and config.

        Returns
        -------
        Mapping[str, int] | None
            None to trigger auto row count computation.
        """
        cfg = self.config
        build_hotspots(ctx.gateway, cfg, runner=ctx.extra.get("tool_runner"))
        return None  # Let base class compute row counts


__all__ = ["HotspotsPlugin"]
