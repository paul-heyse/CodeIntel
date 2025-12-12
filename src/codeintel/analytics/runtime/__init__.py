"""Analytics runtime infrastructure.

This package provides shared runtime infrastructure for analytics modules,
including graph runtime management and manifest encoding.

Modules
-------
- graph: GraphRuntime, GraphRuntimeOptions, and related graph caching infrastructure
- manifest: AnalyticsRunReport and manifest encoding utilities

Example
-------
```python
from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    build_graph_runtime,
    encode_manifest,
)


options = GraphRuntimeOptions(snapshot=snapshot, backend=backend)
runtime = build_graph_runtime(gateway, options)


call_graph = runtime.ensure_call_graph()
```
"""

from __future__ import annotations

from codeintel.analytics.runtime.context import (
    GraphContext,
    GraphContextCaps,
    GraphContextSpec,
    build_graph_context,
    load_prior_manifest,
    resolve_graph_context,
)
from codeintel.analytics.runtime.graph import (
    GraphRuntime,
    GraphRuntimeOptions,
    GraphRuntimePool,
    PooledRuntime,
    build_graph_runtime,
    resolve_graph_runtime,
)
from codeintel.analytics.runtime.manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
    PluginExecutionRecord,
    PluginStatus,
    encode_manifest,
)
from codeintel.graphs.engine import GraphKind

__all__ = [
    "AnalyticsPlanInfo",
    "AnalyticsRunReport",
    "AnalyticsScope",
    "AnalyticsSkippedStep",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphKind",
    "GraphRuntime",
    "GraphRuntimeOptions",
    "GraphRuntimePool",
    "PluginExecutionRecord",
    "PluginStatus",
    "PooledRuntime",
    "build_graph_context",
    "build_graph_runtime",
    "encode_manifest",
    "load_prior_manifest",
    "resolve_graph_context",
    "resolve_graph_runtime",
]
