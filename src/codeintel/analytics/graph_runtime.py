"""Shared graph runtime options for analytics modules."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import networkx as nx
from networkx.readwrite import json_graph

from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags, SnapshotRef
from codeintel.graphs.engine import GraphEngine, GraphKind
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.graphs.nx_backend import BackendEnablement
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


log = logging.getLogger(__name__)


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
    engine: GraphEngine | None = None
    graph_cache_dir: Path | None = None
    features: GraphFeatureFlags = field(default_factory=GraphFeatureFlags)

    @property
    def resolved_backend(self) -> GraphBackendConfig:
        """Return a concrete backend configuration."""
        return self.backend or GraphBackendConfig()

    @property
    def use_gpu(self) -> bool:
        """Compute whether GPU execution is preferred."""
        if self.backend is not None:
            return bool(self.backend.use_gpu)
        return False

    @property
    def resolved_eager(self) -> bool:
        """Eager hydration flag resolved against feature overrides."""
        if self.features.eager_hydration is not None:
            return self.features.eager_hydration
        return self.eager

    def __post_init__(self) -> None:
        """Validate nested feature flags."""
        self.features.validate()


@dataclass
class GraphRuntime:
    """Live runtime wrapping a GraphEngine plus cached graph instances."""

    options: GraphRuntimeOptions
    engine: GraphEngine
    backend_info: BackendEnablement | None = None
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
        graph, cache_hit = self._get_graph(GraphKind.CALL_GRAPH, self.engine.load_call_graph)
        self.call_graph = cast("nx.DiGraph", graph)
        self._log_graph_stats("call_graph", self.call_graph, cache_hit=cache_hit)
        return self.call_graph

    def ensure_import_graph(self) -> nx.DiGraph:
        """
        Return a cached import graph, loading it from the engine when needed.

        Returns
        -------
        nx.DiGraph
            Import graph for the runtime snapshot.
        """
        graph, cache_hit = self._get_graph(GraphKind.IMPORT_GRAPH, self.engine.load_import_graph)
        self.import_graph = cast("nx.DiGraph", graph)
        self._log_graph_stats("import_graph", self.import_graph, cache_hit=cache_hit)
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
        graph, cache_hit = self._get_graph(
            GraphKind.SYMBOL_MODULE_GRAPH, self.engine.load_symbol_module_graph
        )
        self.symbol_module_graph = graph
        self._log_graph_stats("symbol_module_graph", self.symbol_module_graph, cache_hit=cache_hit)
        return self.symbol_module_graph

    def ensure_symbol_function_graph(self) -> nx.Graph:
        """
        Return a cached symbol-function graph, loading from the engine when needed.

        Returns
        -------
        nx.Graph
            Symbol-function coupling graph.
        """
        graph, cache_hit = self._get_graph(
            GraphKind.SYMBOL_FUNCTION_GRAPH, self.engine.load_symbol_function_graph
        )
        self.symbol_function_graph = graph
        self._log_graph_stats(
            "symbol_function_graph", self.symbol_function_graph, cache_hit=cache_hit
        )
        return self.symbol_function_graph

    def ensure_config_module_bipartite(self) -> nx.Graph:
        """
        Return a cached config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Config key to module bipartite graph.
        """
        graph, cache_hit = self._get_graph(
            GraphKind.CONFIG_MODULE_BIPARTITE, self.engine.load_config_module_bipartite
        )
        self.config_module_bipartite = graph
        self._log_graph_stats(
            "config_module_bipartite", self.config_module_bipartite, cache_hit=cache_hit
        )
        return self.config_module_bipartite

    def ensure_test_function_bipartite(self) -> nx.Graph:
        """
        Return a cached test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Test to function bipartite graph.
        """
        graph, cache_hit = self._get_graph(
            GraphKind.TEST_FUNCTION_BIPARTITE, self.engine.load_test_function_bipartite
        )
        self.test_function_bipartite = graph
        self._log_graph_stats(
            "test_function_bipartite", self.test_function_bipartite, cache_hit=cache_hit
        )
        return self.test_function_bipartite

    def _get_graph(
        self,
        kind: GraphKind,
        loader: Callable[[], nx.Graph],
    ) -> tuple[nx.Graph, bool]:
        cache_hit = kind in self._cache
        if cache_hit:
            cached = self._cache[kind]
            return cast("nx.Graph", cached), True
        graph = self._load_with_disk_cache(kind, loader)
        self._cache[kind] = graph
        return graph, False

    def _load_with_disk_cache(
        self,
        kind: GraphKind,
        loader: Callable[[], nx.Graph],
    ) -> nx.Graph:
        if self.options.graph_cache_dir is not None and self.options.snapshot is not None:
            cached = self._read_cached_graph(kind)
            if cached is not None:
                return cached
        graph = loader()
        if self.options.graph_cache_dir is not None and self.options.snapshot is not None:
            self._write_cached_graph(kind, graph)
        return graph

    def _cache_base(self, kind: GraphKind) -> Path:
        if self.options.snapshot is None:
            message = "Snapshot is required for graph cache."
            raise ValueError(message)
        safe_repo = self.options.snapshot.repo.replace("/", "__")
        safe_commit = self.options.snapshot.commit
        raw_name = getattr(kind, "name", None)
        kind_name = raw_name.lower() if isinstance(raw_name, str) else str(kind).lower()
        base = (
            f"{safe_repo}__{safe_commit}"
            f"__{self.backend.backend}__{self.backend.use_gpu}"
            f"__{kind_name}"
        )
        return self.options.graph_cache_dir / base  # type: ignore[operator]

    def _read_cached_graph(self, kind: GraphKind) -> nx.Graph | None:
        base = self._cache_base(kind)
        graph_path = base.with_suffix(".json")
        meta_path = base.with_suffix(".meta")
        if not graph_path.exists() or not meta_path.exists():
            return None
        try:
            lines = meta_path.read_text(encoding="utf-8").splitlines()
            expected_fields = 4
            if len(lines) < expected_fields:
                return None
            repo, commit, backend, use_gpu_str = lines[:4]
            if (
                self.options.snapshot is None
                or repo != self.options.snapshot.repo
                or commit != self.options.snapshot.commit
                or backend != self.backend.backend
                or (use_gpu_str == "true") != self.backend.use_gpu
            ):
                return None
            with graph_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return json_graph.node_link_graph(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _write_cached_graph(self, kind: GraphKind, graph: nx.Graph) -> None:
        base = self._cache_base(kind)
        graph_path = base.with_suffix(".json")
        meta_path = base.with_suffix(".meta")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            serialized = json_graph.node_link_data(graph)
            with graph_path.open("w", encoding="utf-8") as fh:
                json.dump(serialized, fh)
            use_gpu_str = "true" if self.backend.use_gpu else "false"
            meta_path.write_text(
                "\n".join(
                    [
                        self.options.snapshot.repo,  # type: ignore[union-attr]
                        self.options.snapshot.commit,  # type: ignore[union-attr]
                        self.backend.backend,
                        use_gpu_str,
                    ]
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            return

    def _log_graph_stats(self, name: str, graph: nx.Graph, *, cache_hit: bool) -> None:
        try:
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
        except (RuntimeError, TypeError, AttributeError):
            node_count = -1
            edge_count = -1
        log.info(
            "graph_runtime.ensure.%s nodes=%d edges=%d cache_hit=%s use_gpu=%s backend=%s",
            name,
            node_count,
            edge_count,
            cache_hit,
            self.backend.use_gpu,
            self.backend.backend,
        )


def build_graph_runtime(
    gateway: StorageGateway,
    options: GraphRuntimeOptions,
    *,
    context: AnalyticsContext | None = None,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
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
    env :
        Optional environment mapping mutated by backend selection hooks.
    enabler :
        Optional callback invoked to enable GPU backends (used for testing).

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
    if options.engine is not None:
        engine = options.engine
    else:
        engine = build_graph_engine(
            gateway,
            options.snapshot,
            graph_backend=resolved_backend,
            context=resolved_context,
            env=env,
            enabler=enabler,
        )
    backend_info = getattr(engine, "backend_info", None)
    runtime = GraphRuntime(options=options, engine=engine, backend_info=backend_info)
    log.info(
        "graph_runtime.built snapshot=%s@%s backend=%s use_gpu=%s features=%s",
        options.snapshot.repo if options.snapshot else None,
        options.snapshot.commit if options.snapshot else None,
        resolved_backend.backend,
        resolved_backend.use_gpu,
        options.features,
    )

    if options.resolved_eager:
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
        engine=opts.engine,
        graph_cache_dir=opts.graph_cache_dir,
        features=opts.features,
    )
    if opts.engine is not None:
        return GraphRuntime(options=normalized_options, engine=opts.engine)

    return build_graph_runtime(
        gateway,
        normalized_options,
        context=resolved_context,
    )


@dataclass
class PooledRuntime:
    """Runtime wrapper with timestamps for pooling."""

    runtime: GraphRuntime
    created_at: float
    last_used: float


class GraphRuntimePool:
    """LRU/TTL pool for GraphRuntime instances keyed by snapshot/backend."""

    def __init__(
        self,
        *,
        max_size: int = 4,
        ttl_seconds: float | None = None,
        time_func: Callable[[], float] = time.time,
    ) -> None:
        if max_size <= 0:
            message = "max_size must be positive"
            raise ValueError(message)
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._time = time_func
        self._entries: dict[tuple[object, ...], PooledRuntime] = {}

    def get(
        self,
        gateway: StorageGateway,
        options: GraphRuntimeOptions,
        *,
        context: AnalyticsContext | None = None,
    ) -> GraphRuntime:
        """
        Return a pooled runtime or build and cache when missing/expired.

        Returns
        -------
        GraphRuntime
            Runtime bound to the provided snapshot/backend.

        Raises
        ------
        ValueError
            When ``options.snapshot`` is missing.
        """
        if options.snapshot is None:
            message = "GraphRuntimeOptions.snapshot is required for pooling."
            raise ValueError(message)
        now = self._time()
        key = self._key(options)
        entry = self._entries.get(key)
        if entry is not None and not self._expired(entry, now):
            entry.last_used = now
            return entry.runtime

        runtime = resolve_graph_runtime(gateway, options.snapshot, options, context=context)
        self._evict_lru(now)
        self._entries[key] = PooledRuntime(runtime=runtime, created_at=now, last_used=now)
        return runtime

    def _expired(self, entry: PooledRuntime, now: float) -> bool:
        if self._ttl is None:
            return False
        return (now - entry.last_used) > self._ttl

    def _evict_lru(self, now: float) -> None:
        keys_to_drop = [key for key, entry in self._entries.items() if self._expired(entry, now)]
        for key in keys_to_drop:
            self._entries.pop(key, None)

        while len(self._entries) >= self._max_size:
            oldest_key = min(self._entries.items(), key=lambda item: item[1].last_used)[0]
            self._entries.pop(oldest_key, None)

    @staticmethod
    def _key(options: GraphRuntimeOptions) -> tuple[object, ...]:
        snapshot = options.snapshot
        if snapshot is None:
            message = "GraphRuntimeOptions.snapshot is required for pooling."
            raise ValueError(message)
        backend = options.backend or GraphBackendConfig()
        return (
            snapshot.repo,
            snapshot.commit,
            backend.backend,
            backend.use_gpu,
            backend.strict,
            options.graphs,
            options.eager,
            options.validate,
            options.cache_key,
            options.graph_cache_dir,
            options.features,
        )


__all__ = [
    "GraphKind",
    "GraphRuntime",
    "GraphRuntimeOptions",
    "GraphRuntimePool",
    "PooledRuntime",
    "build_graph_runtime",
    "resolve_graph_runtime",
]
