"""Module affinity graph construction and clustering utilities."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import FilterExprContext

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

MIN_SHARED_MODULES = 2
DEFAULT_IMPORT_WEIGHT = 1.0
DEFAULT_SYMBOL_WEIGHT = 0.5
DEFAULT_CONFIG_WEIGHT = 0.3


@dataclass(frozen=True)
class AffinityWeights:
    """Weight configuration for module affinity graph construction."""

    import_weight: float = DEFAULT_IMPORT_WEIGHT
    symbol_weight: float = DEFAULT_SYMBOL_WEIGHT
    config_weight: float = DEFAULT_CONFIG_WEIGHT


@dataclass(frozen=True)
class AffinityFrames:
    """Frames used to compute subsystem affinity."""

    import_graph_edges_frame: pa.Table | None = None
    symbol_use_edges_frame: pa.Table | None = None
    config_values_frame: pa.Table | None = None
    modules_frame: pa.Table | None = None


@dataclass(frozen=True)
class AffinityContext:
    """Shared context for affinity graph construction."""

    modules: set[str]
    repo: str
    commit: str
    weights: AffinityWeights


def load_modules_from_frame(
    modules_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> tuple[set[str], dict[str, list[str]]]:
    """Load modules and tags for subsystem inference.

    Returns
    -------
    tuple[set[str], dict[str, list[str]]]
        Set of module names and tag mappings keyed by module.
    """
    if modules_frame is None or modules_frame.num_rows == 0:
        return set(), {}
    filtered = _rows_for_snapshot(modules_frame, repo=repo, commit=commit)
    modules: set[str] = set()
    tags_by_module: dict[str, list[str]] = {}
    for row in filtered:
        module = row.get("module")
        if module is None:
            continue
        module_name = str(module)
        modules.add(module_name)
        parsed_tags = parse_tags(row.get("tags"))
        if parsed_tags:
            tags_by_module[module_name] = parsed_tags
    return modules, tags_by_module


def parse_tags(raw: object) -> list[str]:
    """
    Normalize tags to a list of strings.

    Returns
    -------
    list[str]
        Parsed tag values.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [raw]
    if isinstance(raw, list):
        return [str(tag) for tag in raw]
    return [str(raw)]


def _add_import_edges(
    graph: RxGraphStore,
    ctx: AffinityContext,
    frame: pa.Table | None,
) -> None:
    if frame is None or frame.num_rows == 0:
        return
    edges_filtered = _rows_for_snapshot(frame, repo=ctx.repo, commit=ctx.commit)
    for row in edges_filtered:
        src = row.get("src_module")
        dst = row.get("dst_module")
        if src is None or dst is None:
            continue
        src_mod = str(src)
        dst_mod = str(dst)
        if src_mod in ctx.modules and dst_mod in ctx.modules:
            add_graph_weight(graph, src_mod, dst_mod, ctx.weights.import_weight)


def _add_symbol_edges(
    graph: RxGraphStore,
    ctx: AffinityContext,
    symbol_use_edges_frame: pa.Table | None,
    modules_frame: pa.Table | None,
) -> None:
    if symbol_use_edges_frame is None or symbol_use_edges_frame.num_rows == 0:
        return
    module_by_path: dict[str, str] = {}
    if modules_frame is not None and modules_frame.num_rows > 0:
        modules_filtered = _rows_for_snapshot(
            modules_frame,
            repo=ctx.repo,
            commit=ctx.commit,
        )
        for row in modules_filtered:
            path = row.get("path")
            module = row.get("module")
            if isinstance(path, str) and module is not None:
                module_by_path[path] = str(module)
    symbol_filtered = _rows_for_snapshot(
        symbol_use_edges_frame,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    for row in symbol_filtered:
        use_path = row.get("use_path")
        def_path = row.get("def_path")
        if not isinstance(use_path, str) or not isinstance(def_path, str):
            continue
        src_mod = module_by_path.get(use_path)
        dst_mod = module_by_path.get(def_path)
        if src_mod is None or dst_mod is None:
            continue
        if src_mod in ctx.modules and dst_mod in ctx.modules:
            add_graph_weight(graph, src_mod, dst_mod, ctx.weights.symbol_weight)


def _add_config_edges(
    graph: RxGraphStore,
    ctx: AffinityContext,
    config_values_frame: pa.Table | None,
) -> None:
    if config_values_frame is None or config_values_frame.num_rows == 0:
        return
    config_filtered = _rows_for_snapshot(
        config_values_frame,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    for row in config_filtered:
        extras = row.get("extras")
        if isinstance(extras, dict):
            reference_modules = extras.get("reference_modules")
        else:
            reference_modules = row.get("reference_modules")
        modules_list = parse_tags(reference_modules)
        filtered = [module for module in modules_list if module in ctx.modules]
        if len(filtered) < MIN_SHARED_MODULES:
            continue
        edge_weight = ctx.weights.config_weight / max(len(filtered) - 1, 1)
        for idx, left in enumerate(filtered):
            for right in filtered[idx + 1 :]:
                add_graph_weight(graph, left, right, edge_weight)


def build_weighted_adjacency(
    snapshot: SnapshotRef,
    modules: set[str],
    frames: AffinityFrames,
    *,
    weights: AffinityWeights | None = None,
) -> dict[str, dict[str, float]]:
    """
    Return a weighted adjacency mapping for the module affinity graph.

    Returns
    -------
    dict[str, dict[str, float]]
        Weighted adjacency mapping.
    """
    graph = build_weighted_graph(
        snapshot,
        modules,
        frames,
        weights=weights,
    )
    return graph_to_adjacency(graph)


def build_weighted_graph(
    snapshot: SnapshotRef,
    modules: set[str],
    frames: AffinityFrames,
    *,
    weights: AffinityWeights | None = None,
) -> GraphInput:
    """
    Build an undirected weighted graph representing module affinity.

    Returns
    -------
    GraphInput
        Weighted graph of module affinity.
    """
    w = weights or AffinityWeights()
    ctx = AffinityContext(
        modules=modules,
        repo=snapshot.repo,
        commit=snapshot.commit,
        weights=w,
    )
    graph = RxGraphStore.undirected()
    for module in modules:
        graph.ensure_node(module)
    _add_import_edges(
        graph,
        ctx,
        frames.import_graph_edges_frame,
    )
    _add_symbol_edges(
        graph,
        ctx,
        frames.symbol_use_edges_frame,
        frames.modules_frame,
    )
    _add_config_edges(
        graph,
        ctx,
        frames.config_values_frame,
    )

    return graph


def _rows_for_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    context = FilterExprContext(repo=repo, commit=commit)
    filtered = context.apply(frame)
    return list(iter_rows(filtered))


def add_graph_weight(graph: RxGraphStore, left: str, right: str, weight: float) -> None:
    """Accumulate symmetric edge weights on an undirected graph."""
    if left == right or weight <= 0:
        return
    graph.add_weighted_edge(left, right, weight=weight)


def graph_to_adjacency(graph: GraphInput) -> dict[str, dict[str, float]]:
    """
    Return a plain adjacency dict copy from a weighted undirected graph.

    Parameters
    ----------
    graph : GraphInput
        Weighted undirected graph to convert.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping of source -> target -> weight.
    """
    store = ensure_store(graph)
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        src_key = str(src_id)
        dst_key = str(dst_id)
        adjacency[src_key][dst_key] = weight
        adjacency[dst_key][src_key] = weight
    return adjacency


def seed_labels_from_tags(tags_by_module: dict[str, list[str]]) -> dict[str, str]:
    """
    Derive seed labels for label propagation based on module tags.

    Returns
    -------
    dict[str, str]
        Initial labels keyed by module.
    """
    labels: dict[str, str] = {}
    for module, tags in tags_by_module.items():
        if not tags:
            continue
        first = tags[0]
        if first is None:
            continue
        labels[module] = str(first).lower()
    return labels


def label_propagation_nx(
    graph: GraphInput,
    seed_labels: dict[str, str],
    max_iters: int = 20,
) -> dict[str, str]:
    """
    Run label propagation over the weighted affinity graph.

    Returns
    -------
    dict[str, str]
        Module -> label mapping after propagation.
    """
    labels: dict[str, str] = {}
    store = ensure_store(graph)
    nodes = [str(node) for node in store.node_ids()]
    for node in nodes:
        seed = seed_labels.get(node)
        labels[node] = seed if seed is not None else node
    frozen: set[str] = set(seed_labels)
    ordered_nodes = sorted(nodes)
    adjacency = graph_to_adjacency(store)

    for _ in range(max_iters):
        changed = False
        for node in ordered_nodes:
            if node in frozen:
                continue
            weights: dict[str, float] = defaultdict(float)
            for neighbor, weight in adjacency.get(node, {}).items():
                neighbor_label = labels.get(neighbor)
                if neighbor_label is None:
                    continue
                weights[neighbor_label] += weight
            if not weights:
                continue
            best_label = max(weights.items(), key=lambda item: (item[1], item[0]))[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break
    return labels


def reassign_small_clusters(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
    min_size: int,
) -> dict[str, str]:
    """
    Merge undersized clusters into heavier neighbors.

    Returns
    -------
    dict[str, str]
        Updated labels after reassignment.
    """
    if min_size <= 1:
        return labels
    cluster_sizes = cluster_sizes_map(labels)
    stable_labels = {label for label, size in cluster_sizes.items() if size >= min_size}
    if len(stable_labels) == len(cluster_sizes):
        return labels

    new_labels = dict(labels)
    for node, label in labels.items():
        if cluster_sizes.get(label, 0) >= min_size:
            continue
        best_label = best_neighbor_label(node, adjacency, new_labels, stable_labels)
        if best_label is not None:
            new_labels[node] = best_label
    return new_labels


def best_neighbor_label(
    node: str,
    adjacency: dict[str, dict[str, float]],
    labels: dict[str, str],
    allowed_labels: set[str],
) -> str | None:
    """
    Select the best neighbor label for a node based on edge weights.

    Returns
    -------
    str | None
        Chosen label or None when no neighbors qualify.
    """
    weights: dict[str, float] = defaultdict(float)
    for neighbor, weight in adjacency.get(node, {}).items():
        label = labels.get(neighbor)
        if label is None or label not in allowed_labels:
            continue
        current = weights.get(label, 0.0)
        weights[label] = current + weight
    if not weights:
        return None
    return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


def limit_clusters(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
    max_clusters: int | None,
) -> dict[str, str]:
    """
    Reduce the number of clusters to the requested maximum.

    Returns
    -------
    dict[str, str]
        Labels with cluster count limited.
    """
    if max_clusters is None:
        return labels
    clusters = clusters_from_labels(labels)
    if len(clusters) <= max_clusters:
        return labels

    kept = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))[:max_clusters]
    kept_labels = {label for label, _ in kept}
    new_labels = dict(labels)
    for node, label in labels.items():
        if label in kept_labels:
            continue
        best_label = best_neighbor_label(node, adjacency, new_labels, kept_labels)
        if best_label is None:
            best_label = sorted(kept_labels)[0]
        new_labels[node] = best_label
    return new_labels


def clusters_from_labels(labels: dict[str, str]) -> dict[str, list[str]]:
    """
    Group modules by assigned label.

    Returns
    -------
    dict[str, list[str]]
        Mapping of label -> sorted modules.
    """
    clusters: dict[str, list[str]] = defaultdict(list)
    for module, label in labels.items():
        clusters[label].append(module)
    for mods in clusters.values():
        mods.sort()
    return clusters


def cluster_sizes_map(labels: dict[str, str]) -> dict[str, int]:
    """
    Return the size of each cluster label.

    Returns
    -------
    dict[str, int]
        Label -> size mapping.
    """
    sizes: dict[str, int] = defaultdict(int)
    for label in labels.values():
        sizes[label] += 1
    return sizes
