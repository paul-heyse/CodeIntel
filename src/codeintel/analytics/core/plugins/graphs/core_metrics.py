"""Core graph metrics plugin using the new protocol.

This module provides the core graph metrics plugin migrated to the
new unified plugin protocol, demonstrating graph-aware plugin patterns.
"""

from __future__ import annotations

import importlib
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
from codeintel.analytics.core.traits import GraphAwareMixin
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphKind


@dataclass
class CoreGraphMetricsPlugin(GraphAwareMixin):
    """Plugin for computing core graph metrics.

    This plugin analyzes call and import graphs to compute:
    - Centrality metrics (PageRank, betweenness, closeness)
    - Degree metrics (in/out/total)
    - Component membership
    - Sink detection
    """

    _graph_requirements: tuple[GraphKind, ...] = (
        GraphKind.CALL_GRAPH,
        GraphKind.IMPORT_GRAPH,
    )

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="core_graph_metrics",
            description="Core function/module graph metrics (centrality, neighbors, components).",
            stage="graph",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="graph_cfg",
                    type_ref="GraphMetricsStepConfig",
                    required=False,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="graph_metrics_functions",
                    tables=("analytics.graph_metrics_functions",),
                    min_rows=1,
                    required_columns=(
                        "repo",
                        "commit",
                        "node_id",
                        "out_degree",
                        "in_degree",
                    ),
                ),
                PluginOutputSpec(
                    name="graph_metrics_modules",
                    tables=("analytics.graph_metrics_modules",),
                    min_rows=1,
                    required_columns=("repo", "commit", "module", "in_degree", "out_degree"),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.graph_metrics_functions", kind="dataset"),
                PluginCapability(name="analytics.graph_metrics_modules", kind="dataset"),
            ),
            capabilities_required=(),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                requires_gpu=False,
                priority=20,
            ),
            tags=("graphs", "metrics", "centrality"),
        )

    @property
    def output_contracts(self) -> tuple[OutputContractSpec, ...]:
        """Return output contracts for validation."""
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

        # Check for graph runtime
        if not ctx.has_graph_runtime():
            errors.append("Graph runtime is required for graph metrics")

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

        # Import the graph metrics module
        try:
            graph_metrics = importlib.import_module("codeintel.analytics.graphs.graph_metrics")
        except ImportError as e:
            return PluginResult.fail(f"Failed to import graph metrics module: {e}")

        # Get graph runtime
        try:
            runtime = ctx.graph_runtime
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Build dependencies
        deps = graph_metrics.GraphMetricsDeps(
            catalog_provider=ctx.catalog if ctx.has_catalog() else None,
            runtime=runtime,
            analytics_context=ctx.analytics_context if ctx.has_analytics_context() else None,
            filters=None,
        )

        # Build config - use from context or create minimal one
        cfg = ctx.get_optional_config(GraphMetricsStepConfig)
        if cfg is None:
            cfg = GraphMetricsStepConfig(
                snapshot=SnapshotRef(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    repo_root=ctx.snapshot.repo_root,
                )
            )

        try:
            graph_metrics.compute_graph_metrics(ctx.gateway, cfg, deps=deps)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Graph metrics computation failed: {e}")

        return PluginResult.ok(
            row_counts={
                "analytics.graph_metrics_functions": -1,  # Unknown until queried
                "analytics.graph_metrics_modules": -1,
            },
        )


__all__ = [
    "CoreGraphMetricsPlugin",
]
