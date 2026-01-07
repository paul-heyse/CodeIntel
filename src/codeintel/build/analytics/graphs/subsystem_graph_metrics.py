"""Subsystem-level graph metrics derived from the import graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.graphs import centrality_directed
from codeintel.build.analytics.compute.row_builders import (
    SubsystemMetricInputs,
    SubsystemMetricRow,
    build_subsystem_graph_rows,
)
from codeintel.build.analytics.graphs.graph_metrics import build_graph_metric_filters_from_sets
from codeintel.build.graphs.compute.metrics.components import (
    condensation_layers,
    find_strongly_connected,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.build.graphs.runtime.context import GraphContext


def _subsystem_centralities(
    graph: GraphInput,
    ctx: GraphContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if graph_node_count(graph) == 0:
        return {}, {}, {}
    centrality = centrality_directed(graph, ctx)
    return centrality.pagerank, centrality.betweenness, centrality.closeness


def _layer_by_subsystem(subsystem_graph: GraphInput) -> dict[str, int]:
    if graph_node_count(subsystem_graph) == 0:
        return {}
    scc_result = find_strongly_connected(subsystem_graph, compute_condensation=True)
    layers = condensation_layers(subsystem_graph, scc_result)
    return {str(node): int(layer) for node, layer in layers.items()}


def _degree_maps(
    subsystem_graph: GraphInput,
    *,
    weight: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    store = ensure_store(subsystem_graph, weight=weight)
    in_degree: dict[str, float] = {str(node): 0.0 for node in store.node_ids()}
    out_degree: dict[str, float] = {str(node): 0.0 for node in store.node_ids()}
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight_val = edge_weight_from_payload(payload)
        out_degree[str(src_id)] = out_degree.get(str(src_id), 0.0) + weight_val
        in_degree[str(dst_id)] = in_degree.get(str(dst_id), 0.0) + weight_val
    return in_degree, out_degree


def _build_subsystem_graph(
    import_graph: GraphInput,
    membership_rows: list[tuple[str, str]],
    graph_ctx: GraphContext,
) -> RxGraphStore:
    module_to_subsystem: dict[str, str] = {
        str(module): str(subsystem_id) for subsystem_id, module in membership_rows
    }
    subsystem_graph = RxGraphStore.directed()
    for subsystem_id, _ in membership_rows:
        subsystem_graph.ensure_node(str(subsystem_id))

    store = ensure_store(import_graph, weight=graph_ctx.betweenness_weight)
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        src_sub = module_to_subsystem.get(str(src_id))
        dst_sub = module_to_subsystem.get(str(dst_id))
        if src_sub is None or dst_sub is None or src_sub == dst_sub:
            continue
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        subsystem_graph.add_weighted_edge(src_sub, dst_sub, weight=weight)
    return subsystem_graph


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


@dataclass(frozen=True)
class SubsystemGraphMetricInputs:
    """Inputs required to compute subsystem graph metrics rows."""

    repo: str
    commit: str
    import_graph: GraphInput
    membership_rows: Iterable[Mapping[str, object]] | Iterable[tuple[str, str]]
    runtime: GraphRuntimeOptions | None = None
    filters: GraphMetricFilters | None = None


def build_subsystem_graph_metrics_rows(
    inputs: SubsystemGraphMetricInputs,
) -> list[SubsystemMetricRow]:
    """Build subsystem-level condensed import graph metrics rows.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.subsystem_graph_metrics.
    """
    runtime_opts = inputs.runtime or GraphRuntimeOptions()
    membership_list = _normalize_membership_rows(
        inputs.membership_rows,
        repo=inputs.repo,
        commit=inputs.commit,
    )
    if not membership_list:
        return []
    active_filters = inputs.filters or _filters_from_memberships(membership_list)
    graph_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=inputs.repo,
            commit=inputs.commit,
            use_gpu=runtime_opts.use_gpu,
            now=datetime.now(UTC),
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )
    membership_list = active_filters.filter_subsystem_memberships(membership_list)
    if not membership_list:
        return []

    subsystem_graph = _build_subsystem_graph(
        active_filters.filter_import_graph(inputs.import_graph),
        membership_list,
        graph_ctx,
    )
    subsystem_graph = active_filters.filter_subsystem_graph(subsystem_graph)

    if graph_node_count(subsystem_graph) == 0:
        return []

    centralities = _subsystem_centralities(subsystem_graph, graph_ctx)
    layer_by_subsystem = _layer_by_subsystem(subsystem_graph)
    degree_maps = _degree_maps(subsystem_graph, weight=graph_ctx.betweenness_weight)

    return build_subsystem_graph_rows(
        SubsystemMetricInputs(
            repo=inputs.repo,
            commit=inputs.commit,
            in_degree=degree_maps[0],
            out_degree=degree_maps[1],
            pagerank=centralities[0],
            betweenness=centralities[1],
            closeness=centralities[2],
            layer=layer_by_subsystem,
            created_at=graph_ctx.resolved_now(),
        )
    )
