"""Module affinity graph construction and clustering utilities."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import networkx as nx

from codeintel.config import SubsystemsStepConfig
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

MIN_SHARED_MODULES = 2


def load_modules(
    gateway: StorageGateway, cfg: SubsystemsStepConfig
) -> tuple[set[str], dict[str, list[str]]]:
    """
    Load modules and tags for subsystem inference.

    Returns
    -------
    tuple[set[str], dict[str, list[str]]]
        Modules present and tags keyed by module.
    """
    con = gateway.con
    rows = con.execute(
        "SELECT module, tags FROM core.modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    if not rows:
        rows = con.execute("SELECT module, tags FROM core.modules").fetchall()

    modules: set[str] = set()
    tags_by_module: dict[str, list[str]] = {}
    for module, tags in rows:
        if module is None:
            continue
        module_name = str(module)
        modules.add(module_name)
        parsed_tags = parse_tags(tags)
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


def build_weighted_adjacency(
    gateway: StorageGateway, cfg: SubsystemsStepConfig, modules: set[str]
) -> dict[str, dict[str, float]]:
    """
    Return a weighted adjacency mapping for the module affinity graph.

    Returns
    -------
    dict[str, dict[str, float]]
        Weighted adjacency mapping.
    """
    graph = build_weighted_graph(gateway, cfg, modules)
    return graph_to_adjacency(graph)


def build_weighted_graph(
    gateway: StorageGateway, cfg: SubsystemsStepConfig, modules: set[str]
) -> nx.Graph:
    """
    Build an undirected weighted graph representing module affinity.

    Returns
    -------
    nx.Graph
        Weighted graph of module affinity.
    """
    con = gateway.con
    graph = nx.Graph()
    graph.add_nodes_from(modules)

    rows = con.execute(
        "SELECT src_module, dst_module FROM graph.import_graph_edges WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        src_mod = str(src)
        dst_mod = str(dst)
        if src_mod in modules and dst_mod in modules:
            add_graph_weight(graph, src_mod, dst_mod, cfg.import_weight)

    rows = con.execute(
        """
        SELECT m_use.module, m_def.module
        FROM graph.symbol_use_edges su
        LEFT JOIN core.modules m_def ON m_def.path = su.def_path
        LEFT JOIN core.modules m_use ON m_use.path = su.use_path
        WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
        """
    ).fetchall()
    for use_module, def_module in rows:
        src_mod = str(use_module)
        dst_mod = str(def_module)
        if src_mod in modules and dst_mod in modules:
            add_graph_weight(graph, src_mod, dst_mod, cfg.symbol_weight)

    rows = con.execute(
        """
        SELECT reference_modules
        FROM analytics.config_values
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    for (mods_raw,) in rows:
        modules_list = parse_tags(mods_raw)
        filtered = [m for m in modules_list if m in modules]
        if len(filtered) < MIN_SHARED_MODULES:
            continue
        weight = cfg.config_weight / max(len(filtered) - 1, 1)
        for idx, left in enumerate(filtered):
            for right in filtered[idx + 1 :]:
                add_graph_weight(graph, left, right, weight)

    return graph


def add_graph_weight(graph: nx.Graph, left: str, right: str, weight: float) -> None:
    """Accumulate symmetric edge weights on an undirected graph."""
    if left == right or weight <= 0:
        return
    if graph.has_edge(left, right):
        graph[left][right]["weight"] += weight
    else:
        graph.add_edge(left, right, weight=weight)


def graph_to_adjacency(graph: nx.Graph) -> dict[str, dict[str, float]]:
    """
    Return a plain adjacency dict copy from a weighted undirected graph.

    Parameters
    ----------
    graph : nx.Graph
        Weighted undirected graph to convert.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping of source -> target -> weight.
    """
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for src, dst, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        adjacency[src][dst] = weight
        adjacency[dst][src] = weight
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
    graph: nx.Graph,
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
    for node in graph.nodes:
        fallback = node if isinstance(node, str) else str(node)
        seed = seed_labels.get(node)
        labels[node] = seed if seed is not None else fallback
    frozen: set[str] = set(seed_labels)
    ordered_nodes = sorted(graph.nodes)

    for _ in range(max_iters):
        changed = False
        for node in ordered_nodes:
            if node in frozen:
                continue
            weights: dict[str, float] = defaultdict(float)
            for neighbor, data in graph[node].items():
                neighbor_label = labels.get(neighbor)
                if neighbor_label is None:
                    continue
                weights[neighbor_label] += float(data.get("weight", 1.0))
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
