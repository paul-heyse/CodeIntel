"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from collections.abc import Collection, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.graphs import (
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
)
from codeintel.build.analytics.compute.row_builders import (
    RowBuildContext,
    SymbolMetricInputs,
    build_symbol_function_rows,
    build_symbol_module_rows,
)
from codeintel.build.analytics.graphs.constants import (
    MAX_BETWEENNESS_NODES,
    MAX_COMMUNITY_NODES,
)
from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
from codeintel.build.analytics.graphs.orchestrator import (
    MetricsPipelineConfig,
    MetricsPipelineRequest,
    build_metrics_pipeline_rows,
    build_store_views,
)
from codeintel.build.graphs.builders import (
    build_symbol_function_graph as _build_symbol_function_graph,
)
from codeintel.build.graphs.builders import (
    build_symbol_module_graph as _build_symbol_module_graph,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from codeintel.build.analytics.compute.graphs import (
        CentralityBundle,
        StructuralMetrics,
    )
    from codeintel.build.analytics.graphs.orchestrator import GraphViews
    from codeintel.build.graphs.runtime.context import GraphContext


def build_symbol_module_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
) -> GraphInput:
    """Build an undirected weighted symbol-module graph from use edges.

    Returns
    -------
    GraphInput
        Undirected graph linking modules by symbol coupling.
    """
    return _build_symbol_module_graph(symbol_use_edges, module_by_path)


def build_symbol_function_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
) -> GraphInput:
    """Build an undirected weighted symbol-function graph from use edges.

    Returns
    -------
    GraphInput
        Undirected graph linking functions by symbol coupling.
    """
    return _build_symbol_function_graph(symbol_use_edges)


def _parse_int_node(node: object) -> int | None:
    parsed: int | None = None
    if isinstance(node, bool):
        parsed = int(node)
    elif isinstance(node, int):
        parsed = node
    elif isinstance(node, float):
        parsed = int(node) if node.is_integer() else None
    elif isinstance(node, str):
        value = node.strip()
        if value:
            try:
                parsed = int(value)
            except ValueError:
                parsed = None
    return parsed


def _filter_nodes(graph: GraphInput, allowed: Collection[Hashable]) -> RxGraphStore:
    store = ensure_store(graph)
    if store.is_directed:
        filtered = RxGraphStore.directed(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    else:
        filtered = RxGraphStore.undirected(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    for node_id in store.node_ids():
        if node_id in allowed:
            filtered.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        if src_id not in allowed or dst_id not in allowed:
            continue
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        filtered.add_weighted_edge(src_id, dst_id, weight=weight)
    return filtered


@dataclass(frozen=True)
class SymbolMetricSlices[TNode]:
    """Precomputed slices for symbol graph metrics."""

    centrality: CentralityBundle
    structure: StructuralMetrics
    comp_id: Mapping[TNode, int]
    comp_size: Mapping[TNode, int]
    node_count: int


@dataclass(frozen=True)
class _SymbolMetricRowConfig[TNode]:
    """Configuration for building symbol graph metric rows."""

    graph_name: str
    build_rows: Callable[[SymbolMetricInputs[TNode]], Sequence[tuple[object, ...]]]


@dataclass(frozen=True)
class _SymbolMetricPipelineConfig[TNode]:
    """Configuration for symbol graph metric pipelines."""

    table_key: str
    graph_name: str
    filter_node: Callable[[object, set[TNode]], bool]
    build_rows: Callable[[SymbolMetricInputs[TNode]], Sequence[tuple[object, ...]]]


@dataclass(frozen=True)
class _SymbolMetricPipelineRequest[TNode]:
    """Request parameters for symbol graph metric pipelines."""

    repo: str
    commit: str
    graph: GraphInput
    known_nodes: set[TNode] | None = None
    runtime: GraphRuntimeOptions | None = None


def _filter_known_nodes[TNode](
    graph: GraphInput,
    *,
    known_nodes: set[TNode] | None,
    filter_node: Callable[[object, set[TNode]], bool],
) -> GraphInput:
    if known_nodes is None:
        return graph
    store = ensure_store(graph)
    allowed = {node for node in store.node_ids() if filter_node(node, known_nodes)}
    return _filter_nodes(store, allowed)


def _build_symbol_context(
    runtime: GraphRuntimeOptions,
    repo: str,
    commit: str,
) -> GraphContext:
    return resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime.use_gpu,
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime.features.community_detection_limit,
        )
    )


def _symbol_metric_slices[TNode](
    views: GraphViews,
    ctx: GraphContext,
    *,
    _node_type: type[TNode] | None = None,
) -> SymbolMetricSlices[TNode]:
    graph = views.undirected
    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(
        graph,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    comp_id, comp_size = component_ids_undirected(graph)
    node_count = graph_node_count(graph)
    return SymbolMetricSlices(
        centrality=centrality,
        structure=structure,
        comp_id=comp_id,
        comp_size=comp_size,
        node_count=node_count,
    )


def _symbol_metric_rows[TNode](
    row_context: RowBuildContext,
    views: GraphViews,
    slices: SymbolMetricSlices[TNode],
    *,
    row_config: _SymbolMetricRowConfig[TNode],
) -> list[tuple[object, ...]]:
    if slices.node_count == 0:
        log_empty_graph(row_config.graph_name, views.undirected)
        return []
    metric_inputs = SymbolMetricInputs[TNode](
        row_context=row_context,
        centrality={
            "betweenness": slices.centrality.betweenness,
            "closeness": slices.centrality.closeness,
            "eigenvector": slices.centrality.eigenvector,
            "harmonic": slices.centrality.harmonic,
        },
        structure={
            "core_number": slices.structure.core_number,
            "constraint": slices.structure.constraint,
            "effective_size": slices.structure.effective_size,
            "community_id": (
                slices.structure.community_id if slices.node_count <= MAX_COMMUNITY_NODES else {}
            ),
        },
        comp_id=slices.comp_id,
        comp_size=slices.comp_size,
    )
    return list(row_config.build_rows(metric_inputs))


def _run_symbol_metric_rows[TNode](
    *,
    config: _SymbolMetricPipelineConfig[TNode],
    request: _SymbolMetricPipelineRequest[TNode],
) -> list[tuple[object, ...]]:
    runtime_opts = request.runtime or GraphRuntimeOptions()
    row_config = _SymbolMetricRowConfig(
        graph_name=config.graph_name,
        build_rows=config.build_rows,
    )
    pipeline_config = MetricsPipelineConfig(
        table_key=config.table_key,
        filter_graph=lambda _filters, input_graph: _filter_known_nodes(
            input_graph,
            known_nodes=request.known_nodes,
            filter_node=config.filter_node,
        ),
        build_context=_build_symbol_context,
        build_views=build_store_views,
        build_slices=_symbol_metric_slices,
        build_rows=lambda repo, commit, ctx, views, slices: _symbol_metric_rows(
            RowBuildContext.from_repo_commit(repo, commit, created_at=ctx.resolved_now()),
            views,
            slices,
            row_config=row_config,
        ),
    )
    pipeline_request = MetricsPipelineRequest(
        repo=request.repo,
        commit=request.commit,
        graph=request.graph,
        runtime=runtime_opts,
        filters=GraphMetricFilters(),
    )
    return build_metrics_pipeline_rows(pipeline_config, pipeline_request)


def _filter_module_node(node: object, known: set[str]) -> bool:
    return str(node) in known


def _filter_function_node(node: object, known: set[int]) -> bool:
    """Check if a function node should be included in the graph.

    Returns
    -------
    bool
        True if the node is in the set of known functions.
    """
    parsed = _parse_int_node(node)
    return parsed is not None and parsed in known


def build_symbol_graph_metrics_module_rows(
    *,
    repo: str,
    commit: str,
    graph: GraphInput,
    known_modules: set[str] | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_modules rows from module symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_modules.
    """
    return _run_symbol_metric_rows(
        config=_SymbolMetricPipelineConfig(
            table_key="analytics.symbol_graph_metrics_modules",
            graph_name="symbol_module_graph",
            filter_node=_filter_module_node,
            build_rows=build_symbol_module_rows,
        ),
        request=_SymbolMetricPipelineRequest(
            repo=repo,
            commit=commit,
            graph=graph,
            known_nodes=known_modules,
            runtime=runtime,
        ),
    )


def build_symbol_graph_metrics_function_rows(
    *,
    repo: str,
    commit: str,
    graph: GraphInput,
    known_functions: set[int] | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_functions rows from function symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_functions.
    """
    return _run_symbol_metric_rows(
        config=_SymbolMetricPipelineConfig(
            table_key="analytics.symbol_graph_metrics_functions",
            graph_name="symbol_function_graph",
            filter_node=_filter_function_node,
            build_rows=build_symbol_function_rows,
        ),
        request=_SymbolMetricPipelineRequest(
            repo=repo,
            commit=commit,
            graph=graph,
            known_nodes=known_functions,
            runtime=runtime,
        ),
    )
