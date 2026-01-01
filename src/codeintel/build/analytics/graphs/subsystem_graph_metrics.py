"""Subsystem-level graph metrics derived from the import graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import networkx as nx

from codeintel.build.analytics.compute.graphs import centrality_directed
from codeintel.build.analytics.compute.row_builders import (
    SubsystemMetricInputs,
    SubsystemMetricRow,
    build_subsystem_graph_rows,
)
from codeintel.build.analytics.graphs.graph_metrics import build_graph_metric_filters_from_sets
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.build.graphs.runtime.context import GraphContext


def _dag_layers(graph: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {str(node): 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        node_key = str(node)
        base = layers.get(node_key, 0)
        for succ in graph.successors(node):
            succ_key = str(succ)
            layers[succ_key] = max(layers.get(succ_key, 0), base + 1)
    return layers


def _subsystem_centralities(
    graph: nx.DiGraph,
    ctx: GraphContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, {}
    centrality = centrality_directed(graph, ctx)
    return centrality.pagerank, centrality.betweenness, centrality.closeness


def _layer_by_subsystem(subsystem_graph: nx.DiGraph) -> dict[str, int]:
    condensation = nx.condensation(subsystem_graph)
    layers = _dag_layers(condensation)
    scc_index = cast("dict[object, object]", condensation.graph.get("mapping", {}))
    layer_map: dict[str, int] = {}
    for node in subsystem_graph.nodes:
        node_key = str(node)
        comp_idx = scc_index.get(node_key)
        layer_map[node_key] = layers.get(str(comp_idx), 0) if comp_idx is not None else 0
    return layer_map


def _degree_maps(
    subsystem_graph: nx.DiGraph, *, weight: str | None
) -> tuple[dict[str, float], dict[str, float]]:
    in_degree_pairs = cast("Iterable[tuple[str, float]]", subsystem_graph.in_degree(weight=weight))
    out_degree_pairs = cast(
        "Iterable[tuple[str, float]]", subsystem_graph.out_degree(weight=weight)
    )
    return (
        {str(node): float(deg) for node, deg in in_degree_pairs},
        {str(node): float(deg) for node, deg in out_degree_pairs},
    )


def _build_subsystem_graph(
    import_graph: nx.DiGraph, membership_rows: list[tuple[str, str]], graph_ctx: GraphContext
) -> nx.DiGraph:
    module_to_subsystem: dict[str, str] = {
        str(module): str(subsystem_id) for subsystem_id, module in membership_rows
    }
    subsystem_graph = nx.DiGraph()
    subsystem_graph.add_nodes_from({subsystem_id for subsystem_id, _ in membership_rows})

    for src, dst, data in import_graph.edges(data=True):
        src_sub = module_to_subsystem.get(str(src))
        dst_sub = module_to_subsystem.get(str(dst))
        if src_sub is None or dst_sub is None or src_sub == dst_sub:
            continue
        weight = _coerce_edge_weight(data.get(graph_ctx.betweenness_weight or "weight", 1.0))
        if subsystem_graph.has_edge(src_sub, dst_sub):
            attrs = subsystem_graph[src_sub][dst_sub]
            attrs["weight"] = _coerce_edge_weight(attrs.get("weight")) + weight
        else:
            subsystem_graph.add_edge(src_sub, dst_sub, weight=weight)
    return subsystem_graph


def _coerce_edge_weight(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 1.0
    return 1.0


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _normalize_membership_rows(
    rows: Iterable[Mapping[str, object]] | Iterable[tuple[str, str]],
    *,
    repo: str,
    commit: str,
) -> list[tuple[str, str]]:
    memberships: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, tuple):
            subsystem_id, module = row
            memberships.append((str(subsystem_id), str(module)))
            continue
        subsystem_id = row.get("subsystem_id")
        module = row.get("module")
        if subsystem_id is None or module is None:
            continue
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        memberships.append((str(subsystem_id), str(module)))
    return memberships


def _filters_from_memberships(memberships: list[tuple[str, str]]) -> GraphMetricFilters:
    modules = {module for _, module in memberships}
    subsystems = {subsystem_id for subsystem_id, _ in memberships}
    return build_graph_metric_filters_from_sets(modules=modules, subsystems=subsystems)


def build_subsystem_graph_metrics_rows(
    *,
    repo: str,
    commit: str,
    import_graph: nx.DiGraph,
    membership_rows: Iterable[Mapping[str, object]] | Iterable[tuple[str, str]],
    runtime: GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> list[SubsystemMetricRow]:
    """Build subsystem-level condensed import graph metrics rows.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.subsystem_graph_metrics.
    """
    runtime_opts = runtime or GraphRuntimeOptions()
    membership_list = _normalize_membership_rows(membership_rows, repo=repo, commit=commit)
    if not membership_list:
        return []
    active_filters = filters or _filters_from_memberships(membership_list)
    graph_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime_opts.use_gpu,
            now=datetime.now(UTC),
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )
    membership_list = active_filters.filter_subsystem_memberships(membership_list)
    if not membership_list:
        return []

    subsystem_graph = _build_subsystem_graph(
        active_filters.filter_import_graph(import_graph),
        membership_list,
        graph_ctx,
    )
    subsystem_graph = active_filters.filter_subsystem_graph(subsystem_graph)

    if subsystem_graph.number_of_nodes() == 0:
        return []

    centralities = _subsystem_centralities(subsystem_graph, graph_ctx)
    layer_by_subsystem = _layer_by_subsystem(subsystem_graph)
    degree_maps = _degree_maps(subsystem_graph, weight=graph_ctx.betweenness_weight)

    return build_subsystem_graph_rows(
        SubsystemMetricInputs(
            repo=repo,
            commit=commit,
            in_degree=degree_maps[0],
            out_degree=degree_maps[1],
            pagerank=centralities[0],
            betweenness=centralities[1],
            closeness=centralities[2],
            layer=layer_by_subsystem,
            created_at=graph_ctx.resolved_now(),
        )
    )
