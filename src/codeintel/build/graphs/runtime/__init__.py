"""Graph runtime public API.

This package is the canonical home for graph runtime concerns:
runtime options, pooling, and graph-context normalization for metric compute.
"""

from __future__ import annotations

from codeintel.build.graphs.engine import GraphKind
from codeintel.build.graphs.runtime.context import (
    DEFAULT_BETWEENNESS_SAMPLE,
    GraphContext,
    GraphContextCaps,
    GraphContextSpec,
    GraphMetricsOptions,
    build_graph_context,
    load_prior_manifest,
    resolve_graph_context,
)
from codeintel.build.graphs.runtime.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    GraphRuntimePool,
    PooledRuntime,
    build_graph_runtime,
    resolve_graph_runtime,
)

__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphKind",
    "GraphMetricsOptions",
    "GraphRuntime",
    "GraphRuntimeOptions",
    "GraphRuntimePool",
    "PooledRuntime",
    "build_graph_context",
    "build_graph_runtime",
    "load_prior_manifest",
    "resolve_graph_context",
    "resolve_graph_runtime",
]
