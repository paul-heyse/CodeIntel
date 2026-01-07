"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import networkx as nx

from codeintel.build.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.build.analytics.compute.row_builders import (
    ModuleMetricExtInputs,
    RowBuildContext,
    build_module_metric_ext_rows,
)
from codeintel.build.analytics.graphs.constants import (
    CENTRALITY_SAMPLE_LIMIT,
    RICH_CLUB_PERCENTILE,
)
from codeintel.build.analytics.graphs.context_helpers import GraphContextFactory
from codeintel.build.analytics.graphs.orchestrator import (
    ExtendedMetricsConfig,
    ExtendedMetricsRequest,
    build_extended_metrics_rows,
)
from codeintel.build.graphs.runtime.context import GraphContext

if TYPE_CHECKING:
    from codeintel.build.analytics.compute.graphs import (
        CentralityBundle,
        ComponentBundle,
        StructuralMetrics,
    )
    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.build.analytics.graphs.orchestrator import (
        GraphViews,
    )
    from codeintel.build.graphs.runtime import GraphRuntimeOptions


@dataclass(frozen=True)
class ModuleGraphSlices:
    """Precomputed graph statistics used for module metrics."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    degree_map: dict[str, int]
    degree_cutoff: int


def _rich_club_cutoff(degree_map: dict[str, int]) -> int:
    """Compute the degree cutoff for rich-club membership.

    Parameters
    ----------
    degree_map
        Mapping of nodes to their degrees.

    Returns
    -------
    int
        Degree threshold for rich-club membership.
    """
    if not degree_map:
        return 0
    sorted_degrees = sorted(degree_map.values(), reverse=True)
    idx = max(0, int(len(sorted_degrees) * RICH_CLUB_PERCENTILE) - 1)
    return sorted_degrees[idx] if idx < len(sorted_degrees) else sorted_degrees[-1]


def _resolve_module_context(runtime: GraphRuntimeOptions, repo: str, commit: str) -> GraphContext:
    """Build graph context with module-specific constants.

    Returns
    -------
    GraphContext
        Graph context configured for module-level metrics.
    """
    return _MODULE_CONTEXT_FACTORY.build(runtime, repo=repo, commit=commit)


def _module_metric_slices(views: GraphViews, ctx: GraphContext) -> ModuleGraphSlices:
    """Compute metric slices for module-level extended metrics.

    Parameters
    ----------
    views
        Graph views containing directed, simplified, and undirected graphs.
    ctx
        Graph context with computation parameters.

    Returns
    -------
    ModuleGraphSlices
        Precomputed statistics for row building.
    """
    centralities = centrality_directed(views.simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(
        views.undirected,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    components = component_metadata(views.simple_graph)
    degree_view = cast("Iterable[tuple[object, float]]", views.simple_graph.degree)
    degree_map: dict[str, int] = {str(node): int(deg) for node, deg in degree_view}
    return ModuleGraphSlices(
        centralities=centralities,
        structure=structure,
        components=components,
        degree_map=degree_map,
        degree_cutoff=_rich_club_cutoff(degree_map),
    )


def _module_metric_rows(
    repo: str,
    commit: str,
    ctx: GraphContext,
    views: GraphViews,
    slices: ModuleGraphSlices,
) -> list[dict[str, object]]:
    """Build rows for module-level extended metrics.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    ctx
        Graph context with computation parameters.
    views
        Graph views for node enumeration.
    slices
        Precomputed metric slices.

    Returns
    -------
    list[dict[str, object]]
        Rows ready for insertion.
    """
    centralities = {
        "betweenness": slices.centralities.betweenness,
        "closeness": slices.centralities.closeness,
        "eigenvector": slices.centralities.eigenvector,
        "harmonic": slices.centralities.harmonic,
    }
    structure = {
        "core_number": slices.structure.core_number,
        "constraint": slices.structure.constraint,
        "effective_size": slices.structure.effective_size,
        "community_id": slices.structure.community_id,
    }
    components = {
        "component_id": slices.components.component_id,
        "component_size": slices.components.component_size,
        "scc_id": slices.components.scc_id,
        "scc_size": slices.components.scc_size,
    }
    nodes = [str(node) for node in views.simple_graph.nodes]
    rich_club = {
        module: slices.degree_map.get(module, 0) >= slices.degree_cutoff
        if slices.degree_cutoff > 0
        else False
        for module in nodes
    }
    row_context = RowBuildContext.from_repo_commit(
        repo,
        commit,
        created_at=ctx.resolved_now(),
    )
    inputs = ModuleMetricExtInputs(
        row_context=row_context,
        ctx=ctx,
        centralities=centralities,
        structure=structure,
        components=components,
        rich_club=rich_club,
        nodes=sorted(nodes),
    )
    return build_module_metric_ext_rows(inputs)


# Context factory for module-level extended metrics
_MODULE_CONTEXT_FACTORY = GraphContextFactory(
    betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
    pagerank_weight="weight",
    betweenness_weight="weight",
)

# Configuration for module-level extended metrics
_MODULE_EXT_CONFIG: ExtendedMetricsConfig[ModuleGraphSlices, dict[str, object]] = (
    ExtendedMetricsConfig(
        table_key="analytics.graph_metrics_modules_ext",
        filter_graph=lambda f, g: f.filter_import_graph(g),
        build_context=_resolve_module_context,
        build_slices=_module_metric_slices,
        build_rows=_module_metric_rows,
    )
)


def build_graph_metrics_modules_ext_rows(
    *,
    repo: str,
    commit: str,
    import_graph: nx.DiGraph,
    runtime: GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> list[dict[str, object]]:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics.

    Parameters
    ----------
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    import_graph
        Import graph to analyze for extended metrics.
    runtime
        Optional runtime options including cached graphs and backend selection.
    filters
        Optional allowlists for restricting graph nodes.

    Returns
    -------
    list[dict[str, object]]
        Rows ready for insertion into analytics.graph_metrics_modules_ext.
    """
    request = ExtendedMetricsRequest(
        repo=repo,
        commit=commit,
        graph=import_graph,
        runtime=runtime,
        filters=filters,
    )
    return build_extended_metrics_rows(_MODULE_EXT_CONFIG, request)
