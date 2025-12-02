"""Build shared analytics artifacts for a repository snapshot."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, cast

import networkx as nx

if TYPE_CHECKING:
    from codeintel.analytics.resources.registry import ResourceRegistry

from codeintel.analytics.ast_features.extract import compute_function_features
from codeintel.analytics.ast_features.model import FunctionAstFeatures
from codeintel.analytics.function_ast_cache import (
    FunctionAst,
    FunctionAstLoadRequest,
    load_function_asts,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    build_graph_runtime,
    resolve_graph_runtime,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.catalog import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsContextConfig:
    """Configuration for constructing an `AnalyticsContext`."""

    repo: str
    commit: str
    repo_root: Path
    catalog_provider: FunctionCatalogProvider | None = None
    max_function_asts: int | None = None
    max_call_graph_nodes: int | None = None
    max_import_graph_nodes: int | None = None
    max_symbol_graph_nodes: int | None = None
    max_symbol_graph_edges: int | None = None
    max_graph_edges: int | None = None
    sample_seed: int = 0
    load_symbol_graphs: bool = False
    metrics_hook: Callable[[dict[str, object]], None] | None = None
    use_gpu: bool = False


@dataclass(frozen=True)
class AnalyticsContextStats:
    """Resource usage and truncation signals for a built context."""

    function_asts: int
    missing_functions: int
    call_graph_nodes: int
    call_graph_edges: int
    import_graph_nodes: int
    import_graph_edges: int
    symbol_module_graph_nodes: int
    symbol_module_graph_edges: int
    symbol_function_graph_nodes: int
    symbol_function_graph_edges: int
    truncated_function_asts: bool
    truncated_call_graph: bool
    truncated_import_graph: bool
    truncated_symbol_module_graph: bool
    truncated_symbol_function_graph: bool


@dataclass(frozen=True)
class AnalyticsResourceCounters:
    """Timing indicators for context construction."""

    catalog_ms: float
    module_map_ms: float
    call_graph_ms: float
    import_graph_ms: float
    function_asts_ms: float
    function_features_ms: float
    symbol_module_graph_ms: float
    symbol_function_graph_ms: float


@dataclass(frozen=True)
class AnalyticsContext:
    """Shared analytics artifacts for a repo/commit snapshot."""

    repo: str
    commit: str
    repo_root: Path
    catalog: FunctionCatalogProvider
    module_map: dict[str, str]
    function_ast_map: dict[int, FunctionAst]
    missing_function_goids: set[int]
    function_features_map: dict[int, FunctionAstFeatures]
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph | None
    symbol_module_graph: nx.Graph | None
    symbol_function_graph: nx.Graph | None
    created_at: datetime
    snapshot_id: str
    stats: AnalyticsContextStats
    resources: AnalyticsResourceCounters
    use_gpu: bool

    @classmethod
    def from_resources(
        cls,
        registry: ResourceRegistry,
        config: AnalyticsContextConfig,
        gateway: StorageGateway,
    ) -> AnalyticsContext:
        """Build AnalyticsContext from resource providers.

        Construct an AnalyticsContext by pulling pre-loaded resources from
        the ResourceRegistry. This is the preferred method for obtaining an
        AnalyticsContext in the new architecture.

        Parameters
        ----------
        registry
            Resource registry containing GraphProvider, CatalogProvider, etc.
        config
            Analytics context configuration.
        gateway
            Storage gateway for loading additional data.

        Returns
        -------
        AnalyticsContext
            The constructed analytics context.

        Examples
        --------
        >>> registry = ResourceRegistry()
        >>> registry.register(GraphProvider, GraphProvider(gateway, snapshot))
        >>> registry.register(CatalogProvider, CatalogProvider(gateway, snapshot))
        >>> ctx = AnalyticsContext.from_resources(registry, config, gateway)
        """
        from codeintel.analytics.resources.asts import AstProvider
        from codeintel.analytics.resources.catalog import CatalogProvider
        from codeintel.analytics.resources.graphs import GraphProvider

        # Get resources from registry
        graph_provider = registry.require(GraphProvider)
        catalog_provider = registry.require(CatalogProvider)

        # Get optional AST provider
        ast_provider = registry.require_or_none(AstProvider)

        # Load catalog
        catalog = catalog_provider.get()

        # Load graphs
        call_graph = graph_provider.call_graph
        import_graph = graph_provider.import_graph
        symbol_module_graph = graph_provider.symbol_module_graph
        symbol_function_graph = graph_provider.symbol_function_graph

        # Load module map
        module_map = load_module_map(gateway, config.repo, config.commit)

        # Load ASTs if provider available
        if ast_provider is not None:
            function_ast_map = ast_provider.function_asts
            missing_function_goids = ast_provider.missing_goids
            function_features_map = ast_provider.function_features
        else:
            function_ast_map = {}
            missing_function_goids = set()
            function_features_map = {}

        # Build stats (simplified - no truncation tracking from providers)
        def _counts(graph: nx.Graph | None) -> tuple[int, int]:
            return (graph.number_of_nodes(), graph.number_of_edges()) if graph else (0, 0)

        stats = AnalyticsContextStats(
            function_asts=len(function_ast_map),
            missing_functions=len(missing_function_goids),
            call_graph_nodes=_counts(call_graph)[0],
            call_graph_edges=_counts(call_graph)[1],
            import_graph_nodes=_counts(import_graph)[0],
            import_graph_edges=_counts(import_graph)[1],
            symbol_module_graph_nodes=_counts(symbol_module_graph)[0],
            symbol_module_graph_edges=_counts(symbol_module_graph)[1],
            symbol_function_graph_nodes=_counts(symbol_function_graph)[0],
            symbol_function_graph_edges=_counts(symbol_function_graph)[1],
            truncated_function_asts=False,
            truncated_call_graph=False,
            truncated_import_graph=False,
            truncated_symbol_module_graph=False,
            truncated_symbol_function_graph=False,
        )

        # Resource counters (not tracked when using providers)
        resource_counters = AnalyticsResourceCounters(
            catalog_ms=0.0,
            module_map_ms=0.0,
            call_graph_ms=0.0,
            import_graph_ms=0.0,
            function_asts_ms=0.0,
            function_features_ms=0.0,
            symbol_module_graph_ms=0.0,
            symbol_function_graph_ms=0.0,
        )

        return cls(
            repo=config.repo,
            commit=config.commit,
            repo_root=config.repo_root,
            catalog=catalog,
            module_map=module_map,
            function_ast_map=function_ast_map,
            missing_function_goids=missing_function_goids,
            function_features_map=function_features_map,
            call_graph=call_graph,
            import_graph=import_graph,
            symbol_module_graph=symbol_module_graph,
            symbol_function_graph=symbol_function_graph,
            created_at=datetime.now(tz=UTC),
            snapshot_id=f"{config.repo}@{config.commit}",
            stats=stats,
            resources=resource_counters,
            use_gpu=config.use_gpu,
        )


def _rotate[T](items: list[T], offset: int) -> list[T]:
    if not items:
        return items
    normalized = offset % len(items)
    return items[normalized:] + items[:normalized]


def _trim_graph(
    graph: nx.Graph,
    *,
    max_nodes: int | None,
    max_edges: int | None,
    seed: int,
    label: str,
) -> tuple[nx.Graph, bool, dict[str, object]]:
    """
    Trim a graph to respect node/edge budgets.

    Returns
    -------
    tuple[nx.Graph, bool, dict[str, object]]
        The trimmed graph, whether truncation occurred, and summary metrics.
    """
    nodes_before = graph.number_of_nodes()
    edges_before = graph.number_of_edges()
    truncated = False
    working: nx.Graph = graph

    if max_nodes is not None and nodes_before > max_nodes:
        truncated = True
        nodes = sorted(graph.nodes, key=str)
        keep_nodes = _rotate(nodes, seed)[:max_nodes]
        working = graph.subgraph(keep_nodes).copy()

    if max_edges is not None and working.number_of_edges() > max_edges:
        truncated = True
        edge_items = sorted(
            working.edges(data=True),
            key=lambda item: (str(item[0]), str(item[1])),
        )
        keep_edges = _rotate(edge_items, seed)[:max_edges]
        trimmed = cast("nx.Graph", working.__class__())
        trimmed.add_nodes_from(working.nodes(data=True))
        trimmed.add_edges_from(keep_edges)
        working = trimmed

    result = {
        "graph": label,
        "nodes_before": nodes_before,
        "edges_before": edges_before,
        "nodes_after": working.number_of_nodes(),
        "edges_after": working.number_of_edges(),
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "seed": seed,
        "truncated": truncated,
    }
    if truncated:
        log.info(
            "Trimmed %s graph nodes=%d->%d edges=%d->%d caps=(%s,%s) seed=%d",
            label,
            nodes_before,
            result["nodes_after"],
            edges_before,
            result["edges_after"],
            max_nodes,
            max_edges,
            seed,
        )
    return working, truncated, result


def _import_graph_or_none(engine: GraphEngine) -> nx.DiGraph | None:
    try:
        graph = engine.load_import_graph()
    except (OSError, RuntimeError, ValueError, nx.NetworkXError):
        log.exception(
            "Failed to load import graph for %s@%s", engine.snapshot.repo, engine.snapshot.commit
        )
        return None
    return graph


def _load_trimmed_graph(
    loader: Callable[[], nx.Graph | None],
    *,
    label: str,
    max_nodes: int | None,
    max_edges: int | None,
    seed: int,
) -> tuple[nx.Graph | None, bool, dict[str, object], float]:
    start = monotonic()
    graph = loader()
    elapsed_ms = (monotonic() - start) * 1000.0
    if graph is None:
        return (
            None,
            False,
            {
                "graph": label,
                "nodes_before": 0,
                "edges_before": 0,
                "nodes_after": 0,
                "edges_after": 0,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "seed": seed,
                "truncated": False,
            },
            elapsed_ms,
        )
    trimmed, truncated, metrics = _trim_graph(
        graph,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed,
        label=label,
    )
    return trimmed, truncated, metrics, elapsed_ms


def _load_graphs_for_context(
    engine: GraphEngine,
    cfg: AnalyticsContextConfig,
) -> tuple[dict[str, nx.Graph | None], dict[str, bool], list[dict[str, object]], dict[str, float]]:
    graphs: dict[str, nx.Graph | None] = {}
    truncated: dict[str, bool] = {}
    graph_metrics: list[dict[str, object]] = []
    timers: dict[str, float] = {}

    def _record(
        label: str,
        loader: Callable[[], nx.Graph | None],
        *,
        max_nodes: int | None,
        max_edges: int | None,
        seed: int,
    ) -> None:
        graph, is_truncated, metrics, elapsed_ms = _load_trimmed_graph(
            loader,
            label=label,
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed=seed,
        )
        graphs[label] = graph
        truncated[label] = is_truncated
        graph_metrics.append(metrics)
        timers[f"{label}_ms"] = elapsed_ms

    _record(
        "call_graph",
        engine.call_graph,
        max_nodes=cfg.max_call_graph_nodes,
        max_edges=cfg.max_graph_edges,
        seed=cfg.sample_seed,
    )
    _record(
        "import_graph",
        lambda: _import_graph_or_none(engine),
        max_nodes=cfg.max_import_graph_nodes,
        max_edges=cfg.max_graph_edges,
        seed=cfg.sample_seed + 1,
    )
    if cfg.load_symbol_graphs:
        _record(
            "symbol_module_graph",
            engine.symbol_module_graph,
            max_nodes=cfg.max_symbol_graph_nodes,
            max_edges=cfg.max_symbol_graph_edges,
            seed=cfg.sample_seed + 2,
        )
        _record(
            "symbol_function_graph",
            engine.symbol_function_graph,
            max_nodes=cfg.max_symbol_graph_nodes,
            max_edges=cfg.max_symbol_graph_edges,
            seed=cfg.sample_seed + 3,
        )
    return graphs, truncated, graph_metrics, timers


def _resolve_engine(
    gateway: StorageGateway,
    cfg: AnalyticsContextConfig,
    runtime: GraphRuntime | GraphRuntimeOptions | None,
    engine: GraphEngine | None,
) -> GraphEngine:
    """
    Normalize runtime inputs and ensure an engine is available.

    Returns
    -------
    GraphEngine
        Active engine derived from the provided runtime or newly constructed.
    """
    normalized_runtime: GraphRuntime | None
    if runtime is None:
        normalized_runtime = None
    elif isinstance(runtime, GraphRuntime):
        normalized_runtime = runtime
    else:
        snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
        normalized_runtime = resolve_graph_runtime(
            gateway,
            snapshot,
            runtime,
        )

    active_engine = engine
    if active_engine is None and normalized_runtime is not None:
        active_engine = normalized_runtime.engine
    if active_engine is None:
        snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
        backend = (
            normalized_runtime.backend
            if normalized_runtime is not None
            else GraphBackendConfig(use_gpu=cfg.use_gpu)
        )
        normalized_runtime = build_graph_runtime(
            gateway,
            GraphRuntimeOptions(
                snapshot=snapshot,
                backend=backend,
                context=None,
            ),
        )
        active_engine = normalized_runtime.engine
    return active_engine


def build_analytics_context(
    gateway: StorageGateway,
    cfg: AnalyticsContextConfig,
    *,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    engine: GraphEngine | None = None,
) -> AnalyticsContext:
    """Construct an `AnalyticsContext` with cached artifacts for a run.

    .. deprecated::
        Use `AnalyticsContextProvider` with `ResourceRegistry` instead.
        This function is deprecated and will be removed in a future version.

    Parameters
    ----------
    gateway
        Storage gateway exposing the DuckDB connection.
    cfg
        Context configuration (repo, commit, budgets).
    runtime
        Optional shared graph runtime used to reuse an existing engine when available.
    engine
        Optional pre-built graph engine to reuse; when omitted the runtime or a freshly
        built runtime will supply one.

    Returns
    -------
    AnalyticsContext
        Shared analytics artifacts scoped to the provided repository snapshot.

    See Also
    --------
    AnalyticsContextProvider : Preferred method for obtaining AnalyticsContext.
    AnalyticsContext.from_resources : Class method to build from ResourceRegistry.
    """
    warnings.warn(
        "build_analytics_context is deprecated. "
        "Use AnalyticsContextProvider with ResourceRegistry instead, "
        "or use AnalyticsContext.from_resources().",
        DeprecationWarning,
        stacklevel=2,
    )
    timers: dict[str, float] = {}
    active_engine = _resolve_engine(
        gateway=gateway,
        cfg=cfg,
        runtime=runtime,
        engine=engine,
    )

    start = monotonic()
    catalog = cfg.catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    timers["catalog_ms"] = (monotonic() - start) * 1000.0

    start = monotonic()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit)
    timers["module_map_ms"] = (monotonic() - start) * 1000.0

    graphs, truncated, graph_metrics, graph_timers = _load_graphs_for_context(active_engine, cfg)
    timers.update(graph_timers)
    if graphs.get("import_graph") is not None:
        graphs["import_graph"] = cast("nx.DiGraph", graphs["import_graph"])
    if "symbol_module_graph_ms" not in timers:
        timers["symbol_module_graph_ms"] = 0.0
        timers["symbol_function_graph_ms"] = 0.0
    graphs.setdefault("symbol_module_graph", None)
    graphs.setdefault("symbol_function_graph", None)
    truncated.setdefault("symbol_module_graph", False)
    truncated.setdefault("symbol_function_graph", False)

    start = monotonic()
    function_ast_map, missing = load_function_asts(
        gateway,
        FunctionAstLoadRequest(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
            catalog_provider=catalog,
            max_functions=cfg.max_function_asts,
        ),
    )
    timers["function_asts_ms"] = (monotonic() - start) * 1000.0

    start = monotonic()
    function_features_map: dict[int, FunctionAstFeatures] = {
        goid: compute_function_features(fn_ast, repo_root=cfg.repo_root)
        for goid, fn_ast in function_ast_map.items()
    }
    timers["function_features_ms"] = (monotonic() - start) * 1000.0

    def _counts(graph: nx.Graph | None) -> tuple[int, int]:
        return (graph.number_of_nodes(), graph.number_of_edges()) if graph is not None else (0, 0)

    stats = AnalyticsContextStats(
        function_asts=len(function_ast_map),
        missing_functions=len(missing),
        call_graph_nodes=_counts(graphs["call_graph"])[0],
        call_graph_edges=_counts(graphs["call_graph"])[1],
        import_graph_nodes=_counts(graphs["import_graph"])[0],
        import_graph_edges=_counts(graphs["import_graph"])[1],
        symbol_module_graph_nodes=_counts(graphs["symbol_module_graph"])[0],
        symbol_module_graph_edges=_counts(graphs["symbol_module_graph"])[1],
        symbol_function_graph_nodes=_counts(graphs["symbol_function_graph"])[0],
        symbol_function_graph_edges=_counts(graphs["symbol_function_graph"])[1],
        truncated_function_asts=(
            cfg.max_function_asts is not None
            and cfg.max_function_asts < len(function_ast_map) + len(missing)
        ),
        truncated_call_graph=truncated["call_graph"],
        truncated_import_graph=truncated["import_graph"],
        truncated_symbol_module_graph=truncated["symbol_module_graph"],
        truncated_symbol_function_graph=truncated["symbol_function_graph"],
    )

    context = AnalyticsContext(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog=catalog,
        module_map=module_map,
        function_ast_map=function_ast_map,
        missing_function_goids=missing,
        function_features_map=function_features_map,
        call_graph=cast("nx.DiGraph", graphs["call_graph"]),
        import_graph=(
            graphs["import_graph"] if isinstance(graphs["import_graph"], nx.DiGraph) else None
        ),
        symbol_module_graph=graphs["symbol_module_graph"],
        symbol_function_graph=graphs["symbol_function_graph"],
        created_at=datetime.now(tz=UTC),
        snapshot_id=f"{cfg.repo}@{cfg.commit}",
        stats=stats,
        resources=AnalyticsResourceCounters(
            catalog_ms=timers["catalog_ms"],
            module_map_ms=timers["module_map_ms"],
            call_graph_ms=timers["call_graph_ms"],
            import_graph_ms=timers["import_graph_ms"],
            function_asts_ms=timers["function_asts_ms"],
            function_features_ms=timers["function_features_ms"],
            symbol_module_graph_ms=timers["symbol_module_graph_ms"],
            symbol_function_graph_ms=timers["symbol_function_graph_ms"],
        ),
        use_gpu=cfg.use_gpu,
    )

    if cfg.metrics_hook is not None:
        cfg.metrics_hook(
            {
                "repo": cfg.repo,
                "commit": cfg.commit,
                "graphs": graph_metrics,
                "timers": timers,
                "truncated": {
                    "call_graph": truncated["call_graph"],
                    "import_graph": truncated["import_graph"],
                    "symbol_module_graph": truncated["symbol_module_graph"],
                    "symbol_function_graph": truncated["symbol_function_graph"],
                },
            }
        )

    log.info(
        (
            "AnalyticsContext built for %s@%s: asts=%d missing=%d "
            "call_graph=%d/%d truncated=%s import_graph=%d/%d truncated=%s"
        ),
        cfg.repo,
        cfg.commit,
        stats.function_asts,
        stats.missing_functions,
        stats.call_graph_nodes,
        stats.call_graph_edges,
        stats.truncated_call_graph,
        stats.import_graph_nodes,
        stats.import_graph_edges,
        stats.truncated_import_graph,
    )
    return context


def ensure_analytics_context(
    gateway: StorageGateway,
    *,
    cfg: AnalyticsContextConfig,
    context: AnalyticsContext | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> AnalyticsContext:
    """
    Return an existing `AnalyticsContext` or build one from the provided config.

    Parameters
    ----------
    gateway:
        Storage gateway exposing the DuckDB connection.
    cfg:
        AnalyticsContextConfig specifying repo, commit, and budgets.
    context:
        Optional pre-built context to reuse.
    runtime:
        Optional graph runtime used to reuse an existing engine and caches.

    Returns
    -------
    AnalyticsContext
        Shared analytics artifacts scoped to the provided repository snapshot.

    Raises
    ------
    ValueError
        If the provided context targets a different repo or commit than `cfg`.
    """
    if context is not None:
        if context.repo != cfg.repo or context.commit != cfg.commit:
            message = (
                "AnalyticsContext mismatch: "
                f"{context.repo}@{context.commit} vs {cfg.repo}@{cfg.commit}"
            )
            raise ValueError(message)
        return context
    if isinstance(runtime, GraphRuntime):
        engine_snapshot = runtime.engine.snapshot
        if engine_snapshot.repo != cfg.repo or engine_snapshot.commit != cfg.commit:
            message = (
                "GraphRuntime mismatch: "
                f"{engine_snapshot.repo}@{engine_snapshot.commit} vs {cfg.repo}@{cfg.commit}"
            )
            raise ValueError(message)
    return build_analytics_context(gateway, cfg, runtime=runtime)
