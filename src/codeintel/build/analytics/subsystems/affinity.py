"""Module affinity graph construction and clustering utilities."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.utilities.snapshot import snapshot_plan
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, stable_key
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, DEFAULT_WEIGHT_POLICY
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.data_models.ids import as_int

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
    rowset = _module_rowset(modules_frame, repo=repo, commit=commit)
    modules: set[str] = set()
    tags_by_module: dict[str, list[str]] = {}
    for row in iter_rows(rowset, ("module", "tags")):
        module = row.get("module")
        if module is None:
            continue
        module_name = str(module)
        modules.add(module_name)
        parsed_tags = _flatten_tags(row.get("tags"))
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


def _import_edge_tuples(
    ctx: AffinityContext,
    frame: pa.Table | None,
) -> list[tuple[str, str, float]]:
    if frame is None or frame.num_rows == 0:
        return []
    edge_table = _import_edge_rowset(frame, repo=ctx.repo, commit=ctx.commit)
    edges: list[tuple[str, str, float]] = []
    for row in iter_rows(edge_table, ("src_module", "dst_module", "edge_count")):
        src = row.get("src_module")
        dst = row.get("dst_module")
        if src is None or dst is None:
            continue
        src_mod = str(src)
        dst_mod = str(dst)
        if src_mod not in ctx.modules or dst_mod not in ctx.modules or src_mod == dst_mod:
            continue
        edge_count = as_int(row.get("edge_count")) or 0
        weight = ctx.weights.import_weight * edge_count
        if weight <= 0:
            continue
        edges.append((src_mod, dst_mod, float(weight)))
    return edges


def _symbol_edge_tuples(
    ctx: AffinityContext,
    symbol_use_edges_frame: pa.Table | None,
    modules_frame: pa.Table | None,
) -> list[tuple[str, str, float]]:
    if symbol_use_edges_frame is None or symbol_use_edges_frame.num_rows == 0:
        return []
    module_lookup = _module_lookup_table(modules_frame, repo=ctx.repo, commit=ctx.commit)
    if module_lookup.num_rows == 0:
        return []
    symbol_edges = _symbol_edge_rowset(
        symbol_use_edges_frame,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    mapped_edges = _symbol_module_edge_table(symbol_edges, module_lookup)
    edges: list[tuple[str, str, float]] = []
    for row in iter_rows(mapped_edges, ("use_module", "def_module", "edge_count")):
        src = row.get("use_module")
        dst = row.get("def_module")
        if src is None or dst is None:
            continue
        src_mod = str(src)
        dst_mod = str(dst)
        if src_mod not in ctx.modules or dst_mod not in ctx.modules or src_mod == dst_mod:
            continue
        edge_count = as_int(row.get("edge_count")) or 0
        weight = ctx.weights.symbol_weight * edge_count
        if weight <= 0:
            continue
        edges.append((src_mod, dst_mod, float(weight)))
    return edges


def _config_edge_tuples(
    ctx: AffinityContext,
    config_values_frame: pa.Table | None,
) -> list[tuple[str, str, float]]:
    if config_values_frame is None or config_values_frame.num_rows == 0:
        return []
    config_rowset = _config_module_rowset(
        config_values_frame,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    edges: list[tuple[str, str, float]] = []
    for row in iter_rows(config_rowset, ("reference_modules",)):
        reference_modules = row.get("reference_modules")
        for raw_modules in _list_values(reference_modules):
            modules_list = _flatten_tags(raw_modules)
            filtered = [module for module in modules_list if module in ctx.modules]
            if len(filtered) < MIN_SHARED_MODULES:
                continue
            edge_weight = ctx.weights.config_weight / max(len(filtered) - 1, 1)
            if edge_weight <= 0:
                continue
            for idx, left in enumerate(filtered):
                for right in filtered[idx + 1 :]:
                    if left == right:
                        continue
                    edges.append((left, right, float(edge_weight)))
    return edges


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
    edges: list[tuple[str, str, float]] = []
    edges.extend(_import_edge_tuples(ctx, frames.import_graph_edges_frame))
    edges.extend(
        _symbol_edge_tuples(
            ctx,
            frames.symbol_use_edges_frame,
            frames.modules_frame,
        )
    )
    edges.extend(_config_edge_tuples(ctx, frames.config_values_frame))
    node_ids = sorted(ctx.modules, key=stable_key)
    spec = EdgeBuildSpec(
        directed=False,
        weight_policy=DEFAULT_WEIGHT_POLICY,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_ids=node_ids,
        node_hint=len(node_ids),
        edge_hint=len(edges),
    )
    return build_store_from_edge_tuples(edges, spec=spec, options=options)


def _module_rowset(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    if "module" not in frame.column_names:
        msg = "Missing module column: module"
        raise ValueError(msg)
    plan = snapshot_plan(frame, repo=repo, commit=commit)
    plan = plan.filter(E.is_valid("module"))
    plan = plan.project(
        {
            "module": E.field("module"),
            "tags": E.field("tags") if "tags" in frame.column_names else E.scalar(None),
        }
    )
    plan = plan.order_by(sort_keys=[("module", "ascending")])
    plan = plan.aggregate(
        keys=[E.field("module")],
        aggregates=[("tags", "list", None, "tags")],
    )
    return materialize_plan(plan, use_threads=True)


def _module_lookup_table(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    if frame is None or frame.num_rows == 0:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=["path", "module"],
        )
    if "path" not in frame.column_names or "module" not in frame.column_names:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=["path", "module"],
        )
    plan = snapshot_plan(frame, repo=repo, commit=commit)
    plan = plan.project(
        {
            "path": E.cast(E.field("path"), "string"),
            "module": E.cast(E.field("module"), "string"),
        }
    )
    plan = plan.filter(E.and_(E.is_valid("path"), E.is_valid("module")))
    plan = plan.aggregate(
        keys=[E.field("path")],
        aggregates=[("module", "min", None, "module")],
    )
    return materialize_plan(plan, use_threads=True)


def _symbol_module_edge_table(
    symbol_edges: pa.Table,
    module_lookup: pa.Table,
) -> pa.Table:
    if symbol_edges.num_rows == 0 or module_lookup.num_rows == 0:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.int64()),
            ],
            names=["use_module", "def_module", "edge_count"],
        )
    symbol_plan = Plan.table(symbol_edges)
    symbol_plan = symbol_plan.project(
        {
            "use_path": E.cast(E.field("use_path"), "string"),
            "def_path": E.cast(E.field("def_path"), "string"),
            "edge_count": E.field("edge_count"),
        }
    )
    symbol_plan = symbol_plan.filter(E.and_(E.is_valid("use_path"), E.is_valid("def_path")))
    module_plan = Plan.table(module_lookup)
    module_plan = module_plan.project(
        {
            "path": E.cast(E.field("path"), "string"),
            "module": E.cast(E.field("module"), "string"),
        }
    )
    module_plan = module_plan.filter(E.and_(E.is_valid("path"), E.is_valid("module")))
    def_join = symbol_plan.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["def_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_path", "edge_count"],
            right_output=["module"],
        ),
    )
    def_join = def_join.project(
        {
            "use_path": E.field("use_path"),
            "def_module": E.field("module"),
            "edge_count": E.field("edge_count"),
        }
    )
    use_join = def_join.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["use_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_module", "edge_count"],
            right_output=["module"],
        ),
    )
    use_join = use_join.project(
        {
            "use_module": E.field("module"),
            "def_module": E.field("def_module"),
            "edge_count": E.field("edge_count"),
        }
    )
    use_join = use_join.filter(E.and_(E.is_valid("use_module"), E.is_valid("def_module")))
    use_join = use_join.filter(E.field("use_module") != E.field("def_module"))
    use_join = use_join.aggregate(
        keys=[E.field("use_module"), E.field("def_module")],
        aggregates=[("edge_count", "sum", None, "edge_count")],
    )
    return materialize_plan(use_join, use_threads=True)


def _symbol_edge_rowset(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    required = {"use_path", "def_path"}
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        msg = f"Missing symbol edge columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(frame, repo=repo, commit=commit, columns=("use_path", "def_path"))
    plan = plan.filter(E.and_(E.is_valid("use_path"), E.is_valid("def_path")))
    plan = plan.aggregate(
        keys=[E.field("use_path"), E.field("def_path")],
        aggregates=[("use_path", "count", None, "edge_count")],
    )
    return materialize_plan(plan, use_threads=True)


def _import_edge_rowset(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    required = {"src_module", "dst_module"}
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        msg = f"Missing import edge columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(frame, repo=repo, commit=commit, columns=("src_module", "dst_module"))
    plan = plan.filter(E.and_(E.is_valid("src_module"), E.is_valid("dst_module")))
    plan = plan.aggregate(
        keys=[E.field("src_module"), E.field("dst_module")],
        aggregates=[("src_module", "count", None, "edge_count")],
    )
    return materialize_plan(plan, use_threads=True)


def _config_module_rowset(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    required = {"config_path", "key", "extras"}
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        msg = f"Missing config reference columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(frame, repo=repo, commit=commit, columns=("config_path", "key", "extras"))
    plan = plan.filter(E.and_(E.is_valid("config_path"), E.is_valid("key")))
    plan = plan.project(
        {
            "config_path": E.field("config_path"),
            "key": E.field("key"),
            "reference_modules": E.field(("extras", "reference_modules")),
        }
    )
    plan = plan.order_by(
        sort_keys=[
            ("config_path", "ascending"),
            ("key", "ascending"),
        ]
    )
    plan = plan.aggregate(
        keys=[E.field("config_path"), E.field("key")],
        aggregates=[("reference_modules", "list", None, "reference_modules")],
    )
    return materialize_plan(plan, use_threads=True)


def _flatten_tags(raw: object) -> list[str]:
    if isinstance(raw, list):
        tags: list[str] = []
        for item in raw:
            tags.extend(parse_tags(item))
        return tags
    return parse_tags(raw)


def _list_values(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def add_graph_weight(graph: RxGraphStore, left: str, right: str, weight: float) -> None:
    """Accumulate symmetric edge weights on an undirected graph."""
    if weight <= 0 or left == right:
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
