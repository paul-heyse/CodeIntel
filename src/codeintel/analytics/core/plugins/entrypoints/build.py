"""Entrypoints plugin using new base classes.

This module detects application entrypoints and maps them to handlers and tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider

from codeintel.analytics.core.base import (
    AnalyticsContextRequiringPlugin,
    ConfiguredTableWriterPlugin,
    GraphRuntimeRequiringPlugin,
)
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginResourceHints,
    PluginStage,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.config.steps_analytics import EntryPointsStepConfig


@dataclass
class EntrypointsPlugin(
    ConfiguredTableWriterPlugin[EntryPointsStepConfig],
    AnalyticsContextRequiringPlugin,
    GraphRuntimeRequiringPlugin,
):
    """Detect HTTP/CLI/job entrypoints and map them to handlers and tests.

    Identifies and maps:
    - HTTP endpoints (Flask, FastAPI, etc.)
    - CLI commands
    - Background job handlers
    - Tests covering each entrypoint
    """

    # Core identification
    plugin_name: ClassVar[str] = "entrypoints.build"
    plugin_stage: ClassVar[PluginStage] = "entrypoints"
    plugin_version: ClassVar[str] = "2.0.0"

    # Configuration binding
    config_type: ClassVar[type[EntryPointsStepConfig]] = EntryPointsStepConfig

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.entrypoints",
        "analytics.entrypoint_tests",
    )

    # Capabilities and dependencies
    provides: ClassVar[tuple[str, ...]] = (
        "analytics.entrypoints",
        "analytics.entrypoint_tests",
    )
    requires: ClassVar[tuple[str, ...]] = ("core.goids",)
    depends_on: ClassVar[tuple[str, ...]] = (
        "subsystems.build",
        "coverage.functions",
        "coverage.test_edges",
        "goids",
    )

    # Categorization
    tags: ClassVar[tuple[str, ...]] = ("entrypoints", "http", "cli")

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        max_runtime_ms=90_000,
        priority=50,
    )

    # Optional requirements
    analytics_context_required: ClassVar[bool] = False
    graph_runtime_required: ClassVar[bool] = False

    def _validate_resource_requirements(  # noqa: PLR6301
        self,
        ctx: PluginExecutionContext,  # noqa: ARG002
    ) -> list[str]:
        """Validate resource requirements.

        Entrypoints plugin has optional context and runtime requirements.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Empty list (requirements are optional).
        """
        return []

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute the entrypoints detection.

        Parameters
        ----------
        ctx
            Execution context with gateway and config.

        Returns
        -------
        Mapping[str, int] | None
            None to trigger auto row count computation.

        Raises
        ------
        RuntimeError
            If AnalyticsContextProvider is not registered.
        """
        cfg = self.config

        # Get required analytics context
        if not ctx.has_resource_by_name("AnalyticsContextProvider"):
            msg = "AnalyticsContextProvider is required"
            raise RuntimeError(msg)
        analytics_provider = cast(
            "AnalyticsContextProvider", ctx.require_by_name("AnalyticsContextProvider")
        )
        analytics_context = analytics_provider.get()

        build_entrypoints(
            ctx.gateway,
            cfg,
            context=analytics_context,
        )
        return None  # Let base class compute row counts


__all__ = ["EntrypointsPlugin"]
