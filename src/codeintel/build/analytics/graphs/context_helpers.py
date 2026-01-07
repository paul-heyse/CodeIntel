"""Context helpers for analytics graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import (
    GraphContext,
    GraphContextSpec,
    resolve_graph_context,
)
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters


@dataclass(frozen=True, slots=True)
class GraphContextFactory:
    """Factory for building GraphContext instances with shared defaults."""

    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"

    def build(
        self,
        runtime: GraphRuntimeOptions,
        *,
        repo: str,
        commit: str,
        overrides: GraphContextOverrides | None = None,
    ) -> GraphContext:
        """Build a GraphContext using runtime defaults with optional overrides.

        Returns
        -------
        GraphContext
            Resolved graph context for the provided repo/commit.
        """
        resolved_overrides = overrides or GraphContextOverrides()
        resolved_use_gpu = (
            runtime.use_gpu if resolved_overrides.use_gpu is None else resolved_overrides.use_gpu
        )
        resolved_limit = (
            runtime.features.community_detection_limit
            if resolved_overrides.community_detection_limit is None
            else resolved_overrides.community_detection_limit
        )
        return resolve_graph_context(
            GraphContextSpec(
                repo=repo,
                commit=commit,
                use_gpu=resolved_use_gpu,
                options=resolved_overrides.options,
                now=datetime.now(UTC),
                betweenness_cap=self.betweenness_cap,
                eigen_cap=self.eigen_cap,
                pagerank_weight=self.pagerank_weight,
                betweenness_weight=self.betweenness_weight,
                community_detection_limit=resolved_limit,
            )
        )


@dataclass(frozen=True, slots=True)
class GraphContextOverrides:
    """Optional overrides applied when building a GraphContext."""

    options: GraphMetricsOptions | None = None
    use_gpu: bool | None = None
    community_detection_limit: int | None = None


@dataclass(frozen=True, slots=True)
class GraphMetricsContext:
    """Aggregated context for graph metrics computation."""

    snapshot: SnapshotRef
    runtime: GraphRuntimeOptions
    graph_context: GraphContext
    filters: GraphMetricFilters

    @classmethod
    def from_inputs(
        cls,
        *,
        snapshot: SnapshotRef,
        runtime: GraphRuntimeOptions | None,
        filters: GraphMetricFilters,
        context_factory: GraphContextFactory,
        overrides: GraphContextOverrides | None = None,
    ) -> GraphMetricsContext:
        """Build GraphMetricsContext from snapshot, runtime, and filters.

        Returns
        -------
        GraphMetricsContext
            Aggregated runtime and filter context for graph metrics.
        """
        runtime_opts = runtime or GraphRuntimeOptions(snapshot=snapshot)
        graph_ctx = context_factory.build(
            runtime_opts,
            repo=snapshot.repo,
            commit=snapshot.commit,
            overrides=overrides,
        )
        return cls(
            snapshot=snapshot,
            runtime=runtime_opts,
            graph_context=graph_ctx,
            filters=filters,
        )


__all__ = ["GraphContextFactory", "GraphContextOverrides", "GraphMetricsContext"]
