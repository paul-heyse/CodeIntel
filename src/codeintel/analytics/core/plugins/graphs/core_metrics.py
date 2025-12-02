"""Core graph metrics plugin using new base classes.

This module computes core function/module graph metrics including
centrality, neighbors, and component information.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from codeintel.analytics.core.base import ConfigBoundPlugin, GraphMetricsPlugin
from codeintel.analytics.core.contracts import OutputContractSpec
from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginResourceHints,
    PluginStage,
)
from codeintel.analytics.core.traits import GraphAwareMixin, WithContractValidation
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphKind


@dataclass
class CoreGraphMetricsPlugin(
    ConfigBoundPlugin[GraphMetricsStepConfig],
    GraphMetricsPlugin,
    GraphAwareMixin,
    WithContractValidation,
):
    """Compute core function/module graph metrics.

    This plugin analyzes call and import graphs to compute:
    - Centrality metrics (PageRank, betweenness, closeness)
    - Degree metrics (in/out/total)
    - Component membership
    - Sink detection
    """

    # Core identification
    plugin_name: ClassVar[str] = "core_graph_metrics"
    plugin_stage: ClassVar[PluginStage] = "graph"
    plugin_version: ClassVar[str] = "2.0.0"

    # Configuration binding (optional)
    config_type: ClassVar[type[GraphMetricsStepConfig]] = GraphMetricsStepConfig
    config_required: ClassVar[bool] = False

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_modules",
    )
    min_rows_per_table: ClassVar[Mapping[str, int]] = {
        "analytics.graph_metrics_functions": 1,
        "analytics.graph_metrics_modules": 1,
    }
    required_columns_per_table: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "analytics.graph_metrics_functions": (
            "repo",
            "commit",
            "node_id",
            "out_degree",
            "in_degree",
        ),
        "analytics.graph_metrics_modules": (
            "repo",
            "commit",
            "module",
            "in_degree",
            "out_degree",
        ),
    }

    # Capabilities
    provides: ClassVar[tuple[str, ...]] = (
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_modules",
    )

    # Categorization
    tags: ClassVar[tuple[str, ...]] = ("graphs", "metrics", "centrality")

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        max_runtime_ms=120_000,
        requires_gpu=False,
        priority=20,
    )

    # Graph requirements
    _graph_requirements: tuple[GraphKind, ...] = (
        GraphKind.CALL_GRAPH,
        GraphKind.IMPORT_GRAPH,
    )

    @property
    def output_contracts(self) -> tuple[OutputContractSpec, ...]:
        """Return explicit output contracts for validation.

        Returns
        -------
        tuple[OutputContractSpec, ...]
            Contracts defining validation rules for output tables.
        """
        return (
            OutputContractSpec(
                table="analytics.graph_metrics_functions",
                min_rows=1,
                required_columns=("repo", "commit", "node_id", "out_degree", "in_degree"),
                description="Function-level graph metrics",
            ),
            OutputContractSpec(
                table="analytics.graph_metrics_modules",
                min_rows=1,
                required_columns=("repo", "commit", "module", "in_degree", "out_degree"),
                description="Module-level graph metrics",
            ),
        )

    def _validate_config_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate config - config is optional for this plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Empty list since config is optional.
        """
        # Config is optional, resolve if available
        if ctx.has_config(GraphMetricsStepConfig):
            self._resolved_config = ctx.get_config(GraphMetricsStepConfig)
        return []

    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
        """Execute the graph metrics computation.

        Parameters
        ----------
        ctx
            Execution context with gateway, runtime, and optional config.

        Returns
        -------
        Mapping[str, int] | None
            None to trigger auto row count computation.
        """
        graph_metrics = importlib.import_module("codeintel.analytics.graphs.graph_metrics")

        runtime = self.get_graph_runtime(ctx)

        # Build dependencies
        deps = graph_metrics.GraphMetricsDeps(
            catalog_provider=self.get_catalog(ctx) if ctx.has_catalog() else None,
            runtime=runtime,
            analytics_context=self.get_analytics_context_or_none(ctx),
            filters=None,
        )

        # Build config - use from context or create minimal one
        cfg = self.get_config_or_none()
        if cfg is None:
            cfg = GraphMetricsStepConfig(
                snapshot=SnapshotRef(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    repo_root=ctx.snapshot.repo_root,
                )
            )

        graph_metrics.compute_graph_metrics(ctx.gateway, cfg, deps=deps)
        return None  # Let base class compute row counts

    def get_analytics_context_or_none(  # noqa: PLR6301
        self, ctx: PluginExecutionContext
    ) -> object | None:
        """Get analytics context or None.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        object | None
            Analytics context or None.
        """
        if ctx.has_analytics_context():
            return ctx.analytics_context
        return None


__all__ = [
    "CoreGraphMetricsPlugin",
]
