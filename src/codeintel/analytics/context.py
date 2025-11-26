"""Build shared analytics artifacts for a repository snapshot."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import cast

import networkx as nx

from codeintel.analytics.function_ast_cache import (
    FunctionAst,
    FunctionAstLoadRequest,
    load_function_asts,
)
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import load_module_map
from codeintel.storage.gateway import StorageGateway

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
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph | None
    symbol_module_graph: nx.Graph | None
    symbol_function_graph: nx.Graph | None
    created_at: datetime
    snapshot_id: str
    stats: AnalyticsContextStats
    resources: AnalyticsResourceCounters
    use_gpu: bool


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


def _import_graph_or_none(
    engine: NxGraphEngine,
) -> nx.DiGraph | None:
    try:
        graph = engine.import_graph()
    except Exception:
        log.exception("Failed to load import graph for %s@%s", engine.repo, engine.commit)
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


def build_analytics_context(
    gateway: StorageGateway,
    cfg: AnalyticsContextConfig,
) -> AnalyticsContext:
    """
    Construct an `AnalyticsContext` with cached artifacts for a run.

    Parameters
    ----------
    gateway
        Storage gateway exposing the DuckDB connection.
    cfg
        Context configuration (repo, commit, budgets).

    Returns
    -------
    AnalyticsContext
        Shared analytics artifacts scoped to the provided repository snapshot.
    """
    timers: dict[str, float] = {}
    graph_metrics: list[dict[str, object]] = []

    start = monotonic()
    catalog = cfg.catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    timers["catalog_ms"] = (monotonic() - start) * 1000.0

    start = monotonic()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit)
    timers["module_map_ms"] = (monotonic() - start) * 1000.0

    engine = NxGraphEngine(
        gateway=gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        use_gpu=cfg.use_gpu,
    )
    graphs: dict[str, nx.Graph | None] = {}
    truncated: dict[str, bool] = {}

    def _record_graph(
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

    _record_graph(
        "call_graph",
        engine.call_graph,
        max_nodes=cfg.max_call_graph_nodes,
        max_edges=cfg.max_graph_edges,
        seed=cfg.sample_seed,
    )

    _record_graph(
        "import_graph",
        lambda: _import_graph_or_none(engine),
        max_nodes=cfg.max_import_graph_nodes,
        max_edges=cfg.max_graph_edges,
        seed=cfg.sample_seed + 1,
    )
    if graphs["import_graph"] is not None:
        graphs["import_graph"] = cast("nx.DiGraph", graphs["import_graph"])

    timers["symbol_module_graph_ms"] = 0.0
    timers["symbol_function_graph_ms"] = 0.0
    graphs["symbol_module_graph"] = None
    graphs["symbol_function_graph"] = None
    truncated["symbol_module_graph"] = False
    truncated["symbol_function_graph"] = False
    if cfg.load_symbol_graphs:
        _record_graph(
            "symbol_module_graph",
            engine.symbol_module_graph,
            max_nodes=cfg.max_symbol_graph_nodes,
            max_edges=cfg.max_symbol_graph_edges,
            seed=cfg.sample_seed + 2,
        )
        _record_graph(
            "symbol_function_graph",
            engine.symbol_function_graph,
            max_nodes=cfg.max_symbol_graph_nodes,
            max_edges=cfg.max_symbol_graph_edges,
            seed=cfg.sample_seed + 3,
        )

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

    def _counts(graph: nx.Graph | None) -> tuple[int, int]:
        return (graph.number_of_nodes(), graph.number_of_edges()) if graph is not None else (0, 0)

    call_nodes, call_edges = _counts(graphs["call_graph"])
    counts = {
        "import": _counts(graphs["import_graph"]),
        "symbol_module": _counts(graphs["symbol_module_graph"]),
        "symbol_function": _counts(graphs["symbol_function_graph"]),
    }

    stats = AnalyticsContextStats(
        function_asts=len(function_ast_map),
        missing_functions=len(missing),
        call_graph_nodes=call_nodes,
        call_graph_edges=call_edges,
        import_graph_nodes=counts["import"][0],
        import_graph_edges=counts["import"][1],
        symbol_module_graph_nodes=counts["symbol_module"][0],
        symbol_module_graph_edges=counts["symbol_module"][1],
        symbol_function_graph_nodes=counts["symbol_function"][0],
        symbol_function_graph_edges=counts["symbol_function"][1],
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
    return build_analytics_context(gateway, cfg)
