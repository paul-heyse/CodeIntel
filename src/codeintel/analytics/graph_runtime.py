"""Shared graph runtime options for analytics modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.engine import GraphEngine, GraphKind
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_service import GraphContext


@dataclass(frozen=True)
class GraphRuntimeOptions:
    """Configuration describing how to construct a `GraphRuntime`."""

    snapshot: SnapshotRef | None = None
    backend: GraphBackendConfig | None = None
    graphs: GraphKind = GraphKind.ALL
    eager: bool = False
    validate: bool = False
    cache_key: str | None = None
    context: AnalyticsContext | None = None
    graph_ctx: GraphContext | None = None
    engine: GraphEngine | None = None

    @property
    def resolved_backend(self) -> GraphBackendConfig:
        """Return a concrete backend configuration."""
        return self.backend or GraphBackendConfig()

    @property
    def use_gpu(self) -> bool:
        """Compute whether GPU execution is preferred."""
        if self.backend is not None:
            return bool(self.backend.use_gpu)
        if self.graph_ctx is not None:
            return bool(self.graph_ctx.use_gpu)
        return False

    def build_engine(
        self,
        gateway: StorageGateway,
        repo: str,
        commit: str,
        *,
        snapshot: SnapshotRef | None = None,
    ) -> GraphEngine:
        """
        Construct or reuse a graph engine for the target snapshot.

        This helper is retained for backward compatibility; prefer
        `resolve_graph_runtime` and `GraphRuntime.engine` in new code.

        Returns
        -------
        GraphEngine
            Engine bound to the resolved snapshot and backend.
        """
        if self.engine is not None:
            return self.engine
        target_snapshot = snapshot or self.snapshot
        if target_snapshot is None:
            target_snapshot = SnapshotRef(
                repo=repo,
                commit=commit,
                repo_root=Path(),
            )
        return build_graph_engine(
            gateway,
            target_snapshot,
            graph_backend=self.resolved_backend,
            context=self.context,
        )


@dataclass
class GraphRuntime:
    """Live runtime wrapping a GraphEngine plus cached graph instances."""

    options: GraphRuntimeOptions
    engine: GraphEngine
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    cfg_graph: nx.DiGraph | None = None
    symbol_module_graph: nx.Graph | None = None
    symbol_function_graph: nx.Graph | None = None
    config_module_bipartite: nx.Graph | None = None
    test_function_bipartite: nx.Graph | None = None
    _cache: dict[GraphKind, object] = field(default_factory=dict, repr=False)

    @property
    def backend(self) -> GraphBackendConfig:
        """Resolved backend configuration for this runtime."""
        return self.options.resolved_backend

    @property
    def use_gpu(self) -> bool:
        """Flag indicating whether the backend prefers GPU execution."""
        return self.backend.use_gpu

    def ensure_call_graph(self) -> nx.DiGraph:
        """
        Return a cached call graph, loading it from the engine when needed.

        Returns
        -------
        nx.DiGraph
            Call graph for the runtime snapshot.
        """
        if self.call_graph is None:
            self.call_graph = self.engine.load_call_graph()
        return self.call_graph

    def ensure_import_graph(self) -> nx.DiGraph:
        """
        Return a cached import graph, loading it from the engine when needed.

        Returns
        -------
        nx.DiGraph
            Import graph for the runtime snapshot.
        """
        if self.import_graph is None:
            self.import_graph = self.engine.load_import_graph()
        return self.import_graph

    def ensure_cfg_graph(self) -> nx.DiGraph | None:
        """
        Return a cached CFG graph when available.

        Returns
        -------
        nx.DiGraph | None
            Cached CFG graph when present; otherwise ``None``.
        """
        if self.cfg_graph is not None:
            return self.cfg_graph
        cached = self._cache.get(GraphKind.CFG_GRAPH)
        if cached is not None:
            self.cfg_graph = cached  # type: ignore[assignment]
        return self.cfg_graph

    def ensure_symbol_module_graph(self) -> nx.Graph:
        """
        Return a cached symbol-module graph, loading from the engine when needed.

        Returns
        -------
        nx.Graph
            Symbol-module coupling graph.
        """
        if self.symbol_module_graph is None:
            self.symbol_module_graph = self.engine.load_symbol_module_graph()
        return self.symbol_module_graph

    def ensure_symbol_function_graph(self) -> nx.Graph:
        """
        Return a cached symbol-function graph, loading from the engine when needed.

        Returns
        -------
        nx.Graph
            Symbol-function coupling graph.
        """
        if self.symbol_function_graph is None:
            self.symbol_function_graph = self.engine.load_symbol_function_graph()
        return self.symbol_function_graph

    def ensure_config_module_bipartite(self) -> nx.Graph:
        """
        Return a cached config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Config key to module bipartite graph.
        """
        if self.config_module_bipartite is None:
            self.config_module_bipartite = self.engine.load_config_module_bipartite()
        return self.config_module_bipartite

    def ensure_test_function_bipartite(self) -> nx.Graph:
        """
        Return a cached test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Test to function bipartite graph.
        """
        if self.test_function_bipartite is None:
            self.test_function_bipartite = self.engine.load_test_function_bipartite()
        return self.test_function_bipartite


def build_graph_runtime(
    gateway: StorageGateway,
    options: GraphRuntimeOptions,
    *,
    context: AnalyticsContext | None = None,
) -> GraphRuntime:
    """
    Construct a GraphRuntime bound to a snapshot and backend configuration.

    Parameters
    ----------
    gateway :
        Storage gateway for the snapshot database.
    options :
        Runtime options describing snapshot, backend, and graph flags.
    context :
        Optional analytics context used to seed graph caches.

    Returns
    -------
    GraphRuntime
        Live runtime bound to the provided snapshot.

    Raises
    ------
    ValueError
        If no snapshot is provided on the options.
    """
    if options.snapshot is None:
        message = "GraphRuntimeOptions.snapshot is required to build a runtime."
        raise ValueError(message)
    resolved_context = context or options.context
    resolved_backend = options.resolved_backend
    engine = build_graph_engine(
        gateway,
        options.snapshot,
        graph_backend=resolved_backend,
        context=resolved_context,
    )
    runtime = GraphRuntime(options=options, engine=engine)

    if options.eager:
        if options.graphs & GraphKind.CALL_GRAPH:
            runtime.ensure_call_graph()
        if options.graphs & GraphKind.IMPORT_GRAPH:
            runtime.ensure_import_graph()
        if options.graphs & GraphKind.SYMBOL_MODULE_GRAPH:
            runtime.ensure_symbol_module_graph()
        if options.graphs & GraphKind.SYMBOL_FUNCTION_GRAPH:
            runtime.ensure_symbol_function_graph()
        if options.graphs & GraphKind.CONFIG_MODULE_BIPARTITE:
            runtime.ensure_config_module_bipartite()
        if options.graphs & GraphKind.TEST_FUNCTION_BIPARTITE:
            runtime.ensure_test_function_bipartite()
    return runtime


def resolve_graph_runtime(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions | None,
    *,
    context: AnalyticsContext | None = None,
) -> GraphRuntime:
    """
    Normalize runtime inputs to a concrete `GraphRuntime`.

    Parameters
    ----------
    gateway:
        Storage gateway providing graph sources.
    snapshot:
        Snapshot reference anchoring the runtime.
    runtime:
        Existing runtime or options to materialize one.
    context:
        Optional analytics context used to seed caches.

    Returns
    -------
    GraphRuntime
        Materialized runtime bound to the provided snapshot.
    """
    if isinstance(runtime, GraphRuntime):
        return runtime

    opts = runtime or GraphRuntimeOptions()
    resolved_snapshot = opts.snapshot or snapshot
    resolved_context = context or opts.context

    normalized_options = GraphRuntimeOptions(
        snapshot=resolved_snapshot,
        backend=opts.backend or GraphBackendConfig(),
        graphs=opts.graphs,
        eager=opts.eager,
        validate=opts.validate,
        cache_key=opts.cache_key,
        context=resolved_context,
        graph_ctx=opts.graph_ctx,
        engine=opts.engine,
    )
    if opts.engine is not None:
        return GraphRuntime(options=normalized_options, engine=opts.engine)

    return build_graph_runtime(
        gateway,
        normalized_options,
        context=resolved_context,
    )


__all__ = [
    "GraphKind",
    "GraphRuntime",
    "GraphRuntimeOptions",
    "build_graph_runtime",
    "resolve_graph_runtime",
]
