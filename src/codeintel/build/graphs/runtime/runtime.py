"""Graph runtime options and helpers.

This module provides a hybrid service layer between Hamilton DAG outputs
and NetworkX graph computations backed by Parquet datasets.

Note
----
For pipeline run and step tracking persistence, see `codeintel.storage.tracking`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar, cast

import networkx as nx
import rustworkx as rx

from codeintel.build.graphs.engine import GraphKind
from codeintel.build.graphs.engine.datasets import resolve_dataset_root
from codeintel.build.graphs.engine.factory import EngineBuildOptions, build_graph_engine
from codeintel.build.graphs.rx.convert import networkx_to_rx, rx_to_networkx
from codeintel.build.graphs.rx.serialization import dumps_node_link_json, loads_node_link_json
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.core.options import ValidationOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from codeintel.build.graphs.engine import GraphEngine
    from codeintel.build.graphs.engine.backend import BackendEnablement
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

GRAPH_CACHE_VERSION = "v3"
GraphCacheValue = nx.Graph | nx.DiGraph
GraphT = TypeVar("GraphT", nx.Graph, nx.DiGraph)


def _coerce_int(value: object) -> int | None:
    """Coerce a primitive value into an integer when possible.

    Returns
    -------
    int | None
        Coerced integer value, or None when coercion is not possible.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (bytes, bytearray, str)):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _safe_graph_count(graph: object, *, primary: str, fallback: str) -> int | None:
    """Fetch graph counts from known methods without raising.

    Returns
    -------
    int | None
        Count from the first available method or None on failure.
    """
    for attr in (primary, fallback):
        func = getattr(graph, attr, None)
        if callable(func):
            try:
                value = func()
            except (RuntimeError, TypeError, ValueError):
                return None
            return _coerce_int(value)
    return None


@dataclass(frozen=True)
class GraphRuntimeOptions:
    """Configuration describing how to construct a `GraphRuntime`.

    Implements OptionsProtocol for consistent validation and serialization.
    The `backend` field is derived from the `graph_backend` build config.
    """

    snapshot: SnapshotRef | None = None
    backend: GraphBackendConfig | None = None
    graphs: GraphKind = GraphKind.ALL
    eager: bool = False
    validate: bool = False
    cache_key: str | None = None
    engine: GraphEngine | None = None
    graph_cache_dir: Path | None = None
    dataset_root_dir: Path | None = None
    features: GraphFeatureFlags = field(default_factory=GraphFeatureFlags)

    @classmethod
    def from_parameters(cls, params: Mapping[str, object]) -> GraphRuntimeOptions:
        """Build GraphRuntimeOptions from raw configuration parameters.

        Parameters
        ----------
        params
            Mapping of configuration values (typically from BuildConfig), including
            the `graph_backend` entry used for backend selection.

        Returns
        -------
        GraphRuntimeOptions
            Parsed options with defaults applied.
        """
        return cls(
            snapshot=None,
            backend=cls._parse_backend_param(params),
            graphs=cls._parse_graph_kind_param(params),
            eager=cls._parse_bool_param(params, key="eager", default=False),
            validate=cls._parse_bool_param(params, key="validate", default=False),
            cache_key=cls._parse_str_param(params, key="cache_key"),
            engine=None,
            graph_cache_dir=cls._parse_graph_cache_dir_param(params),
            dataset_root_dir=cls._parse_path_param(params, key="dataset_root_dir"),
            features=cls._parse_graph_features_param(params),
        )

    @classmethod
    def _parse_bool_param(cls, params: Mapping[str, object], *, key: str, default: bool) -> bool:
        value = params.get(key)
        if value is None:
            return default
        return cls._parse_bool(value, key=key)

    @classmethod
    def _parse_str_param(cls, params: Mapping[str, object], *, key: str) -> str | None:
        value = params.get(key)
        if value is None:
            return None
        return cls._parse_str(value, key=key)

    @classmethod
    def _parse_path_param(cls, params: Mapping[str, object], *, key: str) -> Path | None:
        value = params.get(key)
        if value is None:
            return None
        return cls._parse_path(value, key=key)

    @classmethod
    def _parse_graph_cache_dir_param(cls, params: Mapping[str, object]) -> Path | None:
        value = params.get("graph_cache_dir")
        if value is None:
            return None
        return cls._parse_graph_cache_dir(value)

    @classmethod
    def _parse_graph_kind_param(cls, params: Mapping[str, object]) -> GraphKind:
        value = params.get("graphs")
        if value is None:
            return GraphKind.ALL
        return cls._parse_graph_kind(value)

    @classmethod
    def _parse_backend_param(cls, params: Mapping[str, object]) -> GraphBackendConfig | None:
        value = params.get("graph_backend")
        if value is None:
            value = params.get("backend")
        if value is None:
            return None
        return cls._parse_graph_backend(value)

    @classmethod
    def _parse_graph_features_param(cls, params: Mapping[str, object]) -> GraphFeatureFlags:
        value = params.get("features")
        if value is None:
            return GraphFeatureFlags()
        return cls._parse_graph_features(value)

    @staticmethod
    def _parse_bool(value: object, *, key: str) -> bool:
        if isinstance(value, bool):
            return value
        message = f"Expected {key} to be bool, got {type(value)}"
        raise TypeError(message)

    @staticmethod
    def _parse_str(value: object, *, key: str) -> str:
        if isinstance(value, str):
            return value
        message = f"Expected {key} to be str, got {type(value)}"
        raise TypeError(message)

    @staticmethod
    def _parse_path(value: object, *, key: str) -> Path:
        if isinstance(value, str):
            return Path(value)
        if isinstance(value, Path):
            return value
        message = f"Expected {key} to be str|Path, got {type(value)}"
        raise TypeError(message)

    @staticmethod
    def _parse_graph_cache_dir(value: object) -> Path:
        return GraphRuntimeOptions._parse_path(value, key="graph_cache_dir")

    @staticmethod
    def _parse_graph_kind(value: object) -> GraphKind:
        if isinstance(value, GraphKind):
            return value
        if isinstance(value, int):
            return GraphKind(value)
        if isinstance(value, str):
            tokens = [t.strip() for t in value.split(",") if t.strip()]
            if not tokens:
                message = "graphs value must not be empty"
                raise ValueError(message)
            kind = GraphKind.NONE
            for token in tokens:
                name = token.upper()
                try:
                    kind |= GraphKind[name]
                except KeyError as exc:
                    message = f"Unknown GraphKind: {token!r}"
                    raise ValueError(message) from exc
            return kind
        if isinstance(value, list):
            kind = GraphKind.NONE
            for item in value:
                kind |= GraphRuntimeOptions._parse_graph_kind(item)
            return kind
        message = f"Unsupported graphs value type: {type(value)}"
        raise TypeError(message)

    @staticmethod
    def _parse_graph_backend(value: object) -> GraphBackendConfig:
        if isinstance(value, GraphBackendConfig):
            return value
        if isinstance(value, dict):
            try:
                backend = GraphBackendConfig(**value)
            except TypeError as exc:
                message = f"Invalid backend mapping for GraphBackendConfig: {value!r}"
                raise TypeError(message) from exc
            return backend
        message = f"Unsupported backend value type: {type(value)}"
        raise TypeError(message)

    @staticmethod
    def _parse_graph_features(value: object) -> GraphFeatureFlags:
        if isinstance(value, GraphFeatureFlags):
            value.validate()
            return value
        if isinstance(value, dict):
            try:
                flags = GraphFeatureFlags(**value)
            except TypeError as exc:
                message = f"Invalid features mapping for GraphFeatureFlags: {value!r}"
                raise TypeError(message) from exc
            flags.validate()
            return flags
        message = f"Unsupported features value type: {type(value)}"
        raise TypeError(message)

    @property
    def resolved_backend(self) -> GraphBackendConfig:
        """Return a concrete backend configuration derived from graph_backend."""
        backend = self.backend or GraphBackendConfig()
        if backend.engine == "rustworkx" and backend.backend == "cpu" and not backend.use_gpu:
            return backend
        return GraphBackendConfig(
            use_gpu=False,
            backend="cpu",
            strict=backend.strict,
            engine="rustworkx",
        )

    @property
    def use_gpu(self) -> bool:
        """Compute whether GPU execution is preferred."""
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

    def options_validate(self) -> ValidationOutcome:
        """Validate options and return any issues.

        Returns
        -------
        ValidationOutcome
            Validation result with errors/warnings if any.
        """
        errors: list[str] = []

        # Validate feature flags
        try:
            self.features.validate()
        except (ValueError, TypeError) as exc:
            errors.append(f"Feature flags validation failed: {exc}")

        if errors:
            return ValidationOutcome.failure(*errors)
        return ValidationOutcome.success()

    def with_defaults(self, defaults: Self) -> Self:
        """Merge with default values, preferring self's non-None values.

        Parameters
        ----------
        defaults
            Default options to merge from.

        Returns
        -------
        Self
            New options with defaults filled in.
        """
        return type(self)(
            snapshot=self.snapshot if self.snapshot is not None else defaults.snapshot,
            backend=self.backend if self.backend is not None else defaults.backend,
            graphs=self.graphs,
            eager=self.eager if self.eager else defaults.eager,
            validate=self.validate if self.validate else defaults.validate,
            cache_key=self.cache_key if self.cache_key is not None else defaults.cache_key,
            engine=self.engine if self.engine is not None else defaults.engine,
            graph_cache_dir=(
                self.graph_cache_dir
                if self.graph_cache_dir is not None
                else defaults.graph_cache_dir
            ),
            dataset_root_dir=(
                self.dataset_root_dir
                if self.dataset_root_dir is not None
                else defaults.dataset_root_dir
            ),
            features=self.features,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary for logging/debugging.

        Returns
        -------
        dict[str, object]
            Dictionary representation of options.
        """
        result: dict[str, object] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif hasattr(value, "__dict__"):
                result[f.name] = str(value)
            else:
                result[f.name] = value
        return result


def graph_runtime_options_from_env(env: BuildEnv) -> GraphRuntimeOptions:
    """Build GraphRuntimeOptions derived from BuildEnv execution settings.

    Returns
    -------
    GraphRuntimeOptions
        Runtime options derived from the build environment.
    """
    if env.execution_context is None:
        return GraphRuntimeOptions(snapshot=env.snapshot)
    return GraphRuntimeOptions(
        snapshot=env.snapshot,
        backend=env.execution_context.graph_backend,
        features=env.execution_context.graph_features,
    )


@dataclass
class GraphRuntime:
    """Live runtime wrapping a GraphEngine plus cached graph instances.

    This runtime expects Parquet-backed datasets as the graph sources.
    """

    options: GraphRuntimeOptions
    engine: GraphEngine
    backend_info: BackendEnablement | None = None
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    cfg_graph: nx.DiGraph | None = None
    symbol_module_graph: nx.Graph | None = None
    symbol_function_graph: nx.Graph | None = None
    config_module_bipartite: nx.Graph | None = None
    _cache: dict[GraphKind, GraphCacheValue] = field(default_factory=dict, repr=False)

    @property
    def backend(self) -> GraphBackendConfig:
        """Resolved backend configuration for this runtime."""
        return self.options.resolved_backend

    @property
    def use_gpu(self) -> bool:
        """Flag indicating whether the backend prefers GPU execution."""
        return False

    def ensure_call_graph(self) -> nx.DiGraph:
        """Return a cached call graph, loading it from the engine when needed.

        Returns
        -------
        nx.DiGraph
            Call graph for the runtime snapshot.
        """
        graph, cache_hit = self._get_graph(GraphKind.CALL_GRAPH, self.engine.load_call_graph)
        self.call_graph = graph
        self._log_graph_stats("call_graph", self.call_graph, cache_hit=cache_hit)
        return self.call_graph

    def ensure_import_graph(self) -> nx.DiGraph:
        """Return a cached import graph, loading it from the engine when needed.

        Returns
        -------
        nx.DiGraph
            Import graph for the runtime snapshot.
        """
        graph, cache_hit = self._get_graph(GraphKind.IMPORT_GRAPH, self.engine.load_import_graph)
        self.import_graph = graph
        self._log_graph_stats("import_graph", self.import_graph, cache_hit=cache_hit)
        return self.import_graph

    def ensure_cfg_graph(self) -> nx.DiGraph | None:
        """Return a cached CFG graph when available.

        Returns
        -------
        nx.DiGraph | None
            Cached CFG graph when present; otherwise ``None``.
        """
        if self.cfg_graph is not None:
            return self.cfg_graph
        cached = self._cache.get(GraphKind.CFG_GRAPH)
        if isinstance(cached, nx.DiGraph):
            self.cfg_graph = cached
        return self.cfg_graph

    def ensure_symbol_module_graph(self) -> nx.Graph:
        """Return a cached symbol-module graph, loading from the engine when needed.

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
        """Return a cached symbol-function graph, loading from the engine when needed.

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
        """Return a cached config-module bipartite graph.

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

    def _get_graph(
        self,
        kind: GraphKind,
        loader: Callable[[], GraphT],
    ) -> tuple[GraphT, bool]:
        cache_hit = kind in self._cache
        if cache_hit:
            cached = self._cache[kind]
            return cast("GraphT", cached), True
        graph = self._load_with_disk_cache(kind, loader)
        self._cache[kind] = graph
        return graph, False

    def _load_with_disk_cache(
        self,
        kind: GraphKind,
        loader: Callable[[], GraphT],
    ) -> GraphT:
        if self.options.graph_cache_dir is not None and self.options.snapshot is not None:
            cached = self._read_cached_graph(kind)
            if cached is not None:
                return cast("GraphT", cached)
        graph = loader()
        if self.options.graph_cache_dir is not None and self.options.snapshot is not None:
            self._write_cached_graph(kind, graph)
        return graph

    def _cache_base(self, kind: GraphKind) -> Path:
        if self.options.snapshot is None:
            message = "Snapshot is required for graph cache."
            raise ValueError(message)
        if self.options.graph_cache_dir is None:
            message = "Graph cache directory is required for graph cache."
            raise ValueError(message)
        safe_repo = self.options.snapshot.repo.replace("/", "__")
        safe_commit = self.options.snapshot.commit
        raw_name = getattr(kind, "name", None)
        kind_name = raw_name.lower() if isinstance(raw_name, str) else str(kind).lower()
        use_gpu = self.use_gpu
        base = (
            f"{safe_repo}__{safe_commit}"
            f"__{self.backend.backend}__{use_gpu}"
            f"__{self.backend.engine}__{GRAPH_CACHE_VERSION}"
            f"__{kind_name}"
        )
        return self.options.graph_cache_dir / base

    def _read_cached_graph(self, kind: GraphKind) -> nx.Graph | None:
        base = self._cache_base(kind)
        graph_path = base.with_suffix(".json")
        meta_path = base.with_suffix(".meta")
        if not graph_path.exists() or not meta_path.exists():
            return None
        try:
            lines = meta_path.read_text(encoding="utf-8").splitlines()
            expected_fields = 6
            if len(lines) < expected_fields:
                return None
            version, engine, repo, commit, backend, use_gpu_str = lines[:6]
            if version != GRAPH_CACHE_VERSION or engine != self.backend.engine:
                return None
            if (
                self.options.snapshot is None
                or repo != self.options.snapshot.repo
                or commit != self.options.snapshot.commit
                or backend != self.backend.backend
                or (use_gpu_str == "true") != self.use_gpu
            ):
                return None
            payload = graph_path.read_text(encoding="utf-8")
            rx_graph = loads_node_link_json(payload)
            return rx_to_networkx(rx_graph)
        except (OSError, TypeError, ValueError, rx.JSONDeserializationError):
            return None

    def _write_cached_graph(self, kind: GraphKind, graph: nx.Graph) -> None:
        snapshot = self.options.snapshot
        if snapshot is None:
            message = "Snapshot is required for writing graph cache."
            raise ValueError(message)
        base = self._cache_base(kind)
        graph_path = base.with_suffix(".json")
        meta_path = base.with_suffix(".meta")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rx_graph = networkx_to_rx(graph).graph
            payload = dumps_node_link_json(rx_graph)
            graph_path.write_text(payload, encoding="utf-8")
            use_gpu_str = "true" if self.use_gpu else "false"
            meta_path.write_text(
                "\n".join(
                    [
                        GRAPH_CACHE_VERSION,
                        self.backend.engine,
                        snapshot.repo,
                        snapshot.commit,
                        self.backend.backend,
                        use_gpu_str,
                    ]
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError, rx.JSONSerializationError):
            return

    def _log_graph_stats(self, name: str, graph: nx.Graph, *, cache_hit: bool) -> None:
        node_count = _safe_graph_count(
            graph,
            primary="number_of_nodes",
            fallback="num_nodes",
        )
        edge_count = _safe_graph_count(
            graph,
            primary="number_of_edges",
            fallback="num_edges",
        )
        if node_count is None or edge_count is None:
            node_count = -1
            edge_count = -1
        log.info(
            "graph_runtime.ensure.%s nodes=%d edges=%d cache_hit=%s use_gpu=%s "
            "backend=%s engine=%s",
            name,
            node_count,
            edge_count,
            cache_hit,
            self.use_gpu,
            self.backend.backend,
            self.backend.engine,
        )


def build_graph_runtime(
    options: GraphRuntimeOptions,
    *,
    env: MutableMapping[str, str] | None = None,
    enabler: Callable[[], None] | None = None,
) -> GraphRuntime:
    """Construct a GraphRuntime bound to a snapshot and backend configuration.

    Parameters
    ----------
    options
        Runtime options describing snapshot, graph_backend, and graph flags.
    env
        Optional environment mapping mutated by backend selection hooks.
    enabler
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
    resolved_backend = options.resolved_backend
    dataset_root_dir = resolve_dataset_root(options.snapshot, options.dataset_root_dir)
    if options.engine is not None:
        engine = options.engine
    else:
        engine = build_graph_engine(
            snapshot=options.snapshot,
            dataset_root_dir=dataset_root_dir,
            options=EngineBuildOptions(
                graph_backend=resolved_backend,
                env=env,
                enabler=enabler,
            ),
        )
    backend_info = getattr(engine, "backend_info", None)
    runtime = GraphRuntime(options=options, engine=engine, backend_info=backend_info)
    effective_use_gpu = (
        resolved_backend.use_gpu if resolved_backend.engine != "rustworkx" else False
    )
    log.info(
        "graph_runtime.built snapshot=%s@%s backend=%s use_gpu=%s engine=%s features=%s",
        options.snapshot.repo if options.snapshot else None,
        options.snapshot.commit if options.snapshot else None,
        resolved_backend.backend,
        effective_use_gpu,
        resolved_backend.engine,
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
    return runtime


def resolve_graph_runtime(
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions | None,
) -> GraphRuntime:
    """Normalize runtime inputs to a concrete `GraphRuntime`.

    Parameters
    ----------
    snapshot
        Snapshot reference anchoring the runtime.
    runtime
        Existing runtime or options to materialize one.

    Returns
    -------
    GraphRuntime
        Materialized runtime bound to the provided snapshot.
    """
    if isinstance(runtime, GraphRuntime):
        return runtime

    opts = runtime or GraphRuntimeOptions()
    resolved_snapshot = opts.snapshot or snapshot

    normalized_options = GraphRuntimeOptions(
        snapshot=resolved_snapshot,
        backend=opts.backend or GraphBackendConfig(),
        graphs=opts.graphs,
        eager=opts.eager,
        validate=opts.validate,
        cache_key=opts.cache_key,
        engine=opts.engine,
        graph_cache_dir=opts.graph_cache_dir,
        dataset_root_dir=opts.dataset_root_dir,
        features=opts.features,
    )
    if opts.engine is not None:
        return GraphRuntime(options=normalized_options, engine=opts.engine)

    return build_graph_runtime(normalized_options)


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
        options: GraphRuntimeOptions,
    ) -> GraphRuntime:
        """Return a pooled runtime or build and cache when missing/expired.

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

        runtime = resolve_graph_runtime(options.snapshot, options)
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
            options.use_gpu,
            backend.strict,
            backend.engine,
            options.graphs,
            options.eager,
            options.validate,
            options.cache_key,
            options.graph_cache_dir,
            options.dataset_root_dir,
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
