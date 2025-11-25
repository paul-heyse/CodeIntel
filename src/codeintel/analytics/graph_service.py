"""Shared graph metrics utilities for analytics modules."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import networkx as nx
from networkx.algorithms import bipartite, community, structuralholes
from networkx.exception import NetworkXAlgorithmError, PowerIterationFailedConvergence

from codeintel.config.models import GraphMetricsConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphContext:
    """Execution context for graph computations."""

    repo: str
    commit: str
    now: datetime | None = None
    betweenness_sample: int = 500
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"

    def resolved_now(self) -> datetime:
        """
        Return a concrete timestamp, defaulting to UTC now when unset.

        Returns
        -------
        datetime
            Existing timestamp or the current UTC time.
        """
        return self.now or datetime.now(tz=UTC)


DEFAULT_BETWEENNESS_SAMPLE = 500


def build_graph_context(
    cfg: GraphMetricsConfig,
    *,
    now: datetime | None = None,
    betweenness_cap: int | None = None,
    eigen_cap: int | None = None,
) -> GraphContext:
    """
    Construct a GraphContext from GraphMetricsConfig with optional caps.

    Parameters
    ----------
    cfg :
        Graph metrics configuration values.
    now :
        Optional timestamp; defaults to UTC now when omitted.
    betweenness_cap :
        Optional upper bound for betweenness sampling.
    eigen_cap :
        Optional upper bound for eigenvector iterations.

    Returns
    -------
    GraphContext
        Graph context with caps and seeds applied.
    """
    betweenness_sample = cfg.max_betweenness_sample or DEFAULT_BETWEENNESS_SAMPLE
    if betweenness_cap is not None:
        betweenness_sample = min(betweenness_sample, betweenness_cap)
    eigen_max_iter = cfg.eigen_max_iter if eigen_cap is None else min(cfg.eigen_max_iter, eigen_cap)
    return GraphContext(
        repo=cfg.repo,
        commit=cfg.commit,
        now=now,
        betweenness_sample=betweenness_sample,
        eigen_max_iter=eigen_max_iter,
        seed=cfg.seed,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_weight=cfg.betweenness_weight,
    )


@dataclass
class GraphBundle[TGraph: nx.Graph]:
    """Memoized set of graphs loaded by name."""

    ctx: GraphContext
    loaders: dict[str, Callable[[], TGraph]]
    _cache: dict[str, TGraph] = field(default_factory=dict)

    def get(self, name: str) -> TGraph:
        """
        Load a graph by name once, caching the result.

        Parameters
        ----------
        name : str
            Key registered in the loaders mapping.

        Returns
        -------
        nx.Graph
            Graph instance loaded from the configured loader or cache.

        Raises
        ------
        KeyError
            If no loader is registered for the provided name.
        """
        if name not in self.loaders:
            message = f"Graph loader not found: {name}"
            raise KeyError(message)
        if name not in self._cache:
            self._cache[name] = self.loaders[name]()
        return self._cache[name]


@dataclass(frozen=True)
class NeighborStats:
    """Neighbor and edge count summaries for directed graphs."""

    in_neighbors: dict[Any, set[Any]]
    out_neighbors: dict[Any, set[Any]]
    in_counts: dict[Any, int]
    out_counts: dict[Any, int]


@dataclass(frozen=True)
class CentralityBundle:
    """Centrality metrics for a directed graph."""

    pagerank: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]
    eigenvector: dict[Any, float]


@dataclass(frozen=True)
class ComponentBundle:
    """Component metadata for directed graphs."""

    component_id: dict[Any, int]
    component_size: dict[Any, int]
    scc_id: dict[Any, int]
    scc_size: dict[Any, int]
    in_cycle: dict[Any, bool]
    layer: dict[Any, int]


@dataclass(frozen=True)
class ProjectionMetrics:
    """Centrality bundle for projected bipartite graphs."""

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    clustering: dict[Any, float]
    betweenness: dict[Any, float]


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural graph features for undirected graphs."""

    clustering: dict[Any, float]
    triangles: dict[Any, int]
    core_number: dict[Any, int]
    constraint: dict[Any, float]
    effective_size: dict[Any, float]
    community_id: dict[Any, int]


def to_decimal_id(value: int | str | Decimal | None) -> Decimal | None:
    """
    Coerce identifiers to Decimal for DuckDB writes.

    Parameters
    ----------
    value : int | str | Decimal | None
        Identifier value to normalize.

    Returns
    -------
    Decimal | None
        Decimal-backed identifier or None when no value is provided.
    """
    if value is None:
        return None
    return Decimal(int(value))


def normalize_node_id(node: Decimal | float | str | None) -> int | str | None:
    """
    Normalize graph node identifiers for consistent dictionary keys.

    Returns
    -------
    int | str | None
        Integer for numeric nodes (including Decimal or digit-only strings), otherwise
        stringified value; None is preserved.
    """
    if node is None:
        return None
    if isinstance(node, Decimal):
        return int(node)
    if isinstance(node, (int, float)):
        try:
            return int(node)
        except (TypeError, ValueError):
            return None
    if isinstance(node, str) and node.isdigit():
        return int(node)
    return str(node)


def safe_float(value: float | Decimal | str | None) -> float | None:
    """
    Coerce a value to float when possible.

    Parameters
    ----------
    value : float | int | Decimal | str | None
        Input value to convert via float(). None returns None.

    Returns
    -------
    float | None
        Converted float when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_empty_graph(name: str, graph: nx.Graph) -> None:
    """Emit a debug log when a graph has no nodes."""
    if graph.number_of_nodes() == 0:
        log.debug("Graph %s is empty; metrics will be zeroed", name)


def log_projection_skipped(label: str, reason: str, *, nodes: int, graph_nodes: int) -> None:
    """Emit a warning when a projection cannot be computed."""
    log.warning(
        "Skipping %s projection: %s (partition_size=%d graph_nodes=%d)",
        label,
        reason,
        nodes,
        graph_nodes,
    )


def _betweenness_sample(graph: nx.Graph, ctx: GraphContext) -> int | None:
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return None
    if node_count <= ctx.betweenness_sample:
        return None
    return ctx.betweenness_sample


def neighbor_stats(graph: nx.DiGraph, *, weight: str | None = None) -> NeighborStats:
    """
    Accumulate neighbor sets and weighted edge counts.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph containing edges to evaluate.
    weight : str | None, optional
        Edge attribute storing the weight. When None, uses "weight" if present else 1.

    Returns
    -------
    NeighborStats
        Aggregated inbound/outbound neighbor sets and weighted degree counts.
    """
    in_neighbors: dict[Any, set[Any]] = {}
    out_neighbors: dict[Any, set[Any]] = {}
    in_counts: dict[Any, int] = {}
    out_counts: dict[Any, int] = {}

    for src, dst, data in graph.edges(data=True):
        key = "weight" if weight is None else weight
        weight_val = int(data.get(key, 1)) if key is not None else 1
        out_neighbors.setdefault(src, set()).add(dst)
        in_neighbors.setdefault(dst, set()).add(src)
        out_counts[src] = out_counts.get(src, 0) + weight_val
        in_counts[dst] = in_counts.get(dst, 0) + weight_val
    return NeighborStats(
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_counts=in_counts,
        out_counts=out_counts,
    )


def centrality_directed(
    graph: nx.DiGraph,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_eigen: bool = False,
) -> CentralityBundle:
    """
    Compute centrality metrics on a directed graph with shared defaults.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph to evaluate.
    ctx : GraphContext
        Execution context controlling sampling, iteration limits, and seeds.
    weight : str | None, optional
        Edge attribute storing the weight. Defaults to context betweenness/pagerank weight.
    include_eigen : bool, optional
        Whether to compute eigenvector centrality on an undirected view.

    Returns
    -------
    CentralityBundle
        PageRank, betweenness, closeness, harmonic, and optional eigenvector scores.
    """
    betweenness_raw = nx.betweenness_centrality(
        graph,
        weight=ctx.betweenness_weight if weight is None else weight,
        k=_betweenness_sample(graph, ctx),
        seed=ctx.seed,
    )
    betweenness: dict[Any, float] = {node: float(val) for node, val in betweenness_raw.items()}
    closeness: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        closeness = {node: float(val) for node, val in nx.closeness_centrality(graph).items()}
    harmonic: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        harmonic = {node: float(val) for node, val in nx.harmonic_centrality(graph).items()}
    pagerank: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        pagerank = {
            node: float(val)
            for node, val in nx.pagerank(
                graph,
                weight=ctx.pagerank_weight if weight is None else weight,
            ).items()
        }

    eigenvector: dict[Any, float] = {}
    if include_eigen and graph.number_of_nodes() > 0:
        undirected = graph.to_undirected()
        try:
            eigenvector = nx.eigenvector_centrality(
                undirected,
                max_iter=ctx.eigen_max_iter,
                weight=weight,
            )
        except PowerIterationFailedConvergence:
            log.warning("Eigenvector centrality did not converge for graph=%s", graph)
    return CentralityBundle(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigenvector=eigenvector,
    )


def centrality_undirected(
    graph: nx.Graph,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_structural: bool = False,
) -> CentralityBundle:
    """
    Compute centrality metrics on an undirected graph.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph to evaluate.
    ctx : GraphContext
        Execution context controlling sampling, iteration limits, and seeds.
    weight : str | None, optional
        Edge attribute storing the weight. Defaults to "weight".
    include_structural : bool, optional
        Whether to compute additional structural hole metrics.

    Returns
    -------
    CentralityBundle
        PageRank, betweenness, closeness, harmonic, and eigenvector scores.
    """
    betweenness_raw = nx.betweenness_centrality(
        graph,
        weight=ctx.betweenness_weight if weight is None else weight,
        k=_betweenness_sample(graph, ctx),
        seed=ctx.seed,
    )
    betweenness: dict[Any, float] = {node: float(val) for node, val in betweenness_raw.items()}
    closeness: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        closeness = {node: float(val) for node, val in nx.closeness_centrality(graph).items()}
    harmonic: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        harmonic = {node: float(val) for node, val in nx.harmonic_centrality(graph).items()}
    pagerank: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        pagerank = {
            node: float(val)
            for node, val in nx.pagerank(
                graph,
                weight=ctx.pagerank_weight if weight is None else weight,
            ).items()
        }
    eigenvector: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        try:
            eigenvector = nx.eigenvector_centrality(
                graph,
                max_iter=ctx.eigen_max_iter,
                weight=ctx.pagerank_weight if weight is None else weight,
            )
        except PowerIterationFailedConvergence:
            log.warning("Eigenvector centrality did not converge for undirected graph=%s", graph)

    if include_structural and graph.number_of_nodes() > 0:
        try:
            _ = structuralholes.constraint(graph, weight=weight)
        except NetworkXAlgorithmError:
            log.warning("Structural holes calculation failed for graph=%s", graph)
    return CentralityBundle(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigenvector=eigenvector,
    )


def component_metadata(graph: nx.DiGraph) -> ComponentBundle:
    """
    Return weak component, SCC, cycle, and layer metadata.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph from which to derive connectivity metadata.

    Returns
    -------
    ComponentBundle
        Component identifiers, sizes, cycle membership, and condensation layers.
    """
    if graph.number_of_nodes() == 0:
        return ComponentBundle(
            component_id={},
            component_size={},
            scc_id={},
            scc_size={},
            in_cycle={},
            layer={},
        )

    weak_components = list(nx.weakly_connected_components(graph))
    component_id: dict[Any, int] = {
        node: idx for idx, comp in enumerate(weak_components) for node in comp
    }
    component_size: dict[Any, int] = {node: len(comp) for comp in weak_components for node in comp}
    sccs = list(nx.strongly_connected_components(graph))
    scc_id: dict[Any, int] = {node: idx for idx, comp in enumerate(sccs) for node in comp}
    scc_size: dict[Any, int] = {node: len(comp) for idx, comp in enumerate(sccs) for node in comp}
    in_cycle = {node: size > 1 for node, size in scc_size.items()}

    condensation = nx.condensation(graph, sccs)
    layer = _dag_layers(condensation)
    layer_map = {node: layer.get(scc_id[node], 0) for node in graph.nodes}
    return ComponentBundle(
        component_id=component_id,
        component_size=component_size,
        scc_id=scc_id,
        scc_size=scc_size,
        in_cycle=in_cycle,
        layer=layer_map,
    )


def component_ids_undirected(graph: nx.Graph) -> tuple[dict[Any, int], dict[Any, int]]:
    """
    Return component identifiers and sizes for an undirected graph.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph to analyze.

    Returns
    -------
    tuple[dict[Any, int], dict[Any, int]]
        Mapping of node -> component index and node -> component size.
    """
    if graph.number_of_nodes() == 0:
        return {}, {}
    components = list(nx.connected_components(graph))
    comp_id = {node: idx for idx, comp in enumerate(components) for node in comp}
    comp_size = {node: len(comp) for comp in components for node in comp}
    return comp_id, comp_size


def _dag_layers(graph: nx.DiGraph) -> dict[int, int]:
    layers: dict[int, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def projection_metrics(
    bipartite_graph: nx.Graph,
    nodes: Iterable[Any],
    ctx: GraphContext,
    *,
    weight: str | None = None,
) -> ProjectionMetrics:
    """
    Compute weighted projection metrics for a bipartite partition.

    Parameters
    ----------
    bipartite_graph : nx.Graph
        Bipartite graph to project.
    nodes : Iterable[Any]
        Partition nodes to project onto a weighted one-mode graph.
    ctx : GraphContext
        Execution context controlling sampling and seeds.
    weight : str | None, optional
        Edge attribute storing the weight. Defaults to "weight".

    Returns
    -------
    ProjectionMetrics
        Degree, weighted degree, clustering, and betweenness scores for the projection.
    """
    nodes_set = set(nodes)
    if not nodes_set:
        log_projection_skipped(
            "projection",
            "empty partition",
            nodes=len(nodes_set),
            graph_nodes=bipartite_graph.number_of_nodes(),
        )
        return ProjectionMetrics(degree={}, weighted_degree={}, clustering={}, betweenness={})

    proj = bipartite.weighted_projected_graph(bipartite_graph, nodes_set)
    degree_view = proj.degree(weight=None)
    weighted_view = proj.degree(weight=weight)
    degree = {node: int(deg) for node, deg in degree_view}
    weighted_degree = {node: float(deg) for node, deg in weighted_view}
    clustering_val = nx.clustering(proj, weight=weight) if proj.number_of_nodes() > 0 else {}
    clustering = clustering_val if isinstance(clustering_val, dict) else {}
    betweenness = (
        nx.betweenness_centrality(
            proj,
            weight=weight,
            k=_betweenness_sample(proj, ctx),
            seed=ctx.seed,
        )
        if proj.number_of_nodes() > 0
        else {}
    )
    return ProjectionMetrics(
        degree=degree,
        weighted_degree=weighted_degree,
        clustering=clustering,
        betweenness=betweenness,
    )


def community_ids(graph: nx.Graph, *, weight: str | None = "weight") -> dict[Any, int]:
    """
    Assign community ids via asynchronous label propagation.

    Parameters
    ----------
    graph : nx.Graph
        Graph to partition.
    weight : str | None, optional
        Edge attribute storing the weight. Defaults to "weight".

    Returns
    -------
    dict[Any, int]
        Mapping of node to community index.
    """
    if graph.number_of_nodes() == 0:
        return {}
    comms = list(community.asyn_lpa_communities(graph, weight=weight))
    return {node: idx for idx, comm in enumerate(comms) for node in comm}


def structural_metrics(
    graph: nx.Graph,
    *,
    weight: str | None = "weight",
) -> StructuralMetrics:
    """
    Compute undirected structural metrics shared across analytics modules.

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph to evaluate.
    weight : str | None, optional
        Edge attribute storing the weight. Defaults to "weight".

    Returns
    -------
    StructuralMetrics
        Clustering, triangle counts, core numbers, constraint, effective size, and communities.
    """
    if graph.number_of_nodes() == 0:
        return StructuralMetrics(
            clustering={},
            triangles={},
            core_number={},
            constraint={},
            effective_size={},
            community_id={},
        )
    core_number = nx.core_number(graph)
    clustering_val = nx.clustering(graph, weight=weight)
    clustering = clustering_val if isinstance(clustering_val, dict) else {}
    triangles_val = nx.triangles(graph)
    triangles = triangles_val if isinstance(triangles_val, dict) else {}
    constraint_vals = structuralholes.constraint(graph, weight=weight)
    eff_size = structuralholes.effective_size(graph, weight=weight)
    comms = community_ids(graph, weight=weight)
    return StructuralMetrics(
        clustering=clustering,
        triangles=triangles,
        core_number=core_number,
        constraint=constraint_vals,
        effective_size=eff_size,
        community_id=comms,
    )
