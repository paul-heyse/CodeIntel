"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from codeintel.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.analytics.compute.row_builders import (
    ModuleMetricExtInputs,
    build_module_metric_ext_rows,
)
from codeintel.analytics.graphs.constants import (
    CENTRALITY_SAMPLE_LIMIT,
    RICH_CLUB_PERCENTILE,
)
from codeintel.analytics.graphs.orchestrator import (
    ExtendedMetricsConfig,
    ExtendedMetricsRequest,
    build_extended_metrics_rows,
)
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.analytics.compute.graphs import (
        CentralityBundle,
        ComponentBundle,
        StructuralMetrics,
    )
    from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.analytics.graphs.orchestrator import (
        GraphViews,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsGraphMetricsModulesExtRow as GraphMetricsModulesExtRow,
    )
    from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions
    from codeintel.graphs.runtime.context import GraphContext
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class ModuleGraphSlices:
    """Precomputed graph statistics used for module metrics."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    degree_map: dict[object, int]
    degree_cutoff: int


def _rich_club_cutoff(degree_map: dict[object, int]) -> int:
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

    Parameters
    ----------
    runtime
        Runtime options including GPU and community detection settings.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    GraphContext
        Resolved graph context for module metrics computation.
    """
    return resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime.features.community_detection_limit,
        )
    )


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
    degree_map: dict[object, int] = {node: int(deg) for node, deg in degree_view}
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
) -> list[GraphMetricsModulesExtRow]:
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
    list[GraphMetricsModulesExtRow]
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
    rich_club = {
        module: slices.degree_map.get(module, 0) >= slices.degree_cutoff
        if slices.degree_cutoff > 0
        else False
        for module in views.simple_graph.nodes
    }
    inputs = ModuleMetricExtInputs(
        repo=repo,
        commit=commit,
        ctx=ctx,
        centralities=centralities,
        structure=structure,
        components=components,
        rich_club=rich_club,
        nodes=sorted(views.simple_graph.nodes),
    )
    return build_module_metric_ext_rows(inputs)


# Configuration for module-level extended metrics
_MODULE_EXT_CONFIG: ExtendedMetricsConfig[ModuleGraphSlices, GraphMetricsModulesExtRow] = (
    ExtendedMetricsConfig(
        table_key="analytics.graph_metrics_modules_ext",
        get_source_graph=lambda rt: rt.ensure_import_graph(),
        filter_graph=lambda f, g: f.filter_import_graph(g),
        build_context=_resolve_module_context,
        build_slices=_module_metric_slices,
        build_rows=_module_metric_rows,
    )
)


def build_graph_metrics_modules_ext_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> list[GraphMetricsModulesExtRow]:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection used for reads and writes.
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    runtime
        Optional runtime options including cached graphs and backend selection.
    filters
        Optional allowlists for restricting graph nodes.

    Returns
    -------
    list[GraphMetricsModulesExtRow]
        Rows ready for insertion into analytics.graph_metrics_modules_ext.
    """
    request = ExtendedMetricsRequest(
        repo=repo,
        commit=commit,
        runtime=runtime,
        filters=filters,
    )
    return build_extended_metrics_rows(gateway, _MODULE_EXT_CONFIG, request)
