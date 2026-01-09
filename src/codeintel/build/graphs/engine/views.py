"""Shared helpers to materialize Parquet-backed graphs as rustworkx stores.

This module provides functions to load various graph types from
Parquet datasets into rustworkx graph stores. View-registry
fallthrough is intentionally disallowed in this layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import reader_to_table, table_to_reader
from codeintel.build.graphs.engine.datasets import GraphViewFactory, GraphViewScanOptions
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.policies import (
    DEFAULT_NUMERIC_POLICY,
    GraphWeightPolicy,
    weight_policy_for_kind,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.columnar.dedupe_ops import dedupe_keep_first_after_sort
from codeintel.core.columnar.iter import iter_array_values, iter_tuples
from codeintel.core.data_models.ids import as_int
from codeintel.core.data_models.ids import normalize_decimal_id as normalize_decimal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

log = logging.getLogger(__name__)


def _ensure_dataset_root(dataset_root: Path | None, table_key: str) -> Path | None:
    if dataset_root is None:
        log.warning("Dataset root not configured; cannot load %s", table_key)
        return None
    return dataset_root


def _view_factory(
    dataset_root: Path,
    *,
    repo: str | None,
    commit: str,
) -> GraphViewFactory:
    return GraphViewFactory.for_snapshot(dataset_root, repo=repo, commit=commit)


def _column_index(names: list[str], column: str) -> int | None:
    try:
        return names.index(column)
    except ValueError:
        return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _aggregate_edge_counts(
    table: pa.Table,
    *,
    src: str,
    dst: str,
) -> pa.Table:
    if src not in table.column_names or dst not in table.column_names:
        return table
    plan = Plan.table(table)
    plan = plan.filter(E.and_(E.is_valid(src), E.is_valid(dst)))
    plan = plan.project({src: E.field(src), dst: E.field(dst)})
    plan = plan.aggregate(
        keys=[E.field(src), E.field(dst)],
        aggregates=[(src, "count", None, "weight")],
    )
    plan = plan.order_by(sort_keys=[(src, "ascending"), (dst, "ascending")])
    return materialize_plan(plan, use_threads=True)


def _filter_edge_table(
    table: pa.Table,
    *,
    src: str,
    dst: str,
) -> pa.Table:
    if src not in table.column_names or dst not in table.column_names:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=[src, dst],
        )
    plan = Plan.table(table)
    plan = plan.filter(E.and_(E.is_valid(src), E.is_valid(dst)))
    plan = plan.project({src: E.field(src), dst: E.field(dst)})
    plan = plan.order_by(sort_keys=[(src, "ascending"), (dst, "ascending")])
    return materialize_plan(plan, use_threads=True)


def _rename_weight_column(table: pa.Table, *, count_col: str) -> pa.Table:
    if count_col == "weight" or count_col not in table.column_names:
        return table
    if "weight" in table.column_names:
        table = table.drop_columns(["weight"])
    names = ["weight" if name == count_col else name for name in table.column_names]
    return table.rename_columns(names)


def _iter_table_tuples(
    table: pa.Table,
    *,
    columns: Sequence[str],
) -> Iterable[tuple[object, ...]]:
    yield from iter_tuples(table_to_reader(table), columns=columns)


def _node_ids_from_table(
    table: pa.Table,
    *,
    columns: Sequence[str],
    normalize: Callable[[object], Hashable | None],
) -> set[Hashable]:
    node_ids: set[Hashable] = set()
    for column in columns:
        if column not in table.column_names:
            continue
        for value in iter_array_values(table[column]):
            node_id = normalize(value)
            if node_id is not None:
                node_ids.add(node_id)
    return node_ids


@dataclass(frozen=True)
class _EdgeTableSpec:
    src: str
    dst: str
    directed: bool
    weight_policy: GraphWeightPolicy
    normalize: Callable[[object], Hashable | None]
    aggregate_edges: bool = False


def _edge_table_to_store(
    table: pa.Table,
    *,
    spec: _EdgeTableSpec,
    node_ids: Iterable[Hashable] | None = None,
) -> RxGraphStore:
    node_list = list(node_ids) if node_ids is not None else None
    build_spec = EdgeBuildSpec(
        directed=spec.directed,
        weight_policy=spec.weight_policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=spec.normalize,
        dst_fn=spec.normalize,
    )
    return build_store_from_edge_tuples(
        _iter_table_tuples(table, columns=(spec.src, spec.dst, "weight")),
        spec=build_spec,
        options=BuildStoreOptions(
            stable_nodes=True,
            aggregate_edges=spec.aggregate_edges,
            node_ids=node_list,
            node_hint=len(node_list) if node_list is not None else None,
            edge_hint=table.num_rows,
        ),
    )


def _iter_scoped_rows(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> Iterable[tuple[object, ...]]:
    names = list(reader.schema.names)
    repo_idx = _column_index(names, "repo")
    commit_idx = _column_index(names, "commit")
    repo = factory.scan_context.repo
    commit = factory.scan_context.commit
    for row in factory.iter_tuples(reader):
        if repo_idx is not None and repo is not None:
            row_repo = row[repo_idx]
            if row_repo is not None and str(row_repo) != repo:
                continue
        if commit_idx is not None and commit is not None:
            row_commit = row[commit_idx]
            if row_commit is not None and str(row_commit) != commit:
                continue
        yield row


def _empty_graph(*, directed: bool, kind: GraphKind) -> RxGraphStore:
    policy = weight_policy_for_kind(kind)
    return (
        RxGraphStore.directed(weight_policy=policy)
        if directed
        else RxGraphStore.undirected(weight_policy=policy)
    )


def _call_node_attrs(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> dict[int, dict[str, object]]:
    node_attrs: dict[int, dict[str, object]] = {}
    for node_raw, kind in factory.iter_tuples(reader):
        node_id = normalize_decimal(node_raw)
        if node_id is None:
            continue
        attrs = node_attrs.setdefault(node_id, {})
        if kind is not None:
            attrs["kind"] = str(kind)
    return node_attrs


def _presence_flag(
    table: pa.Table,
    *,
    column: str,
) -> pa.Array | pa.ChunkedArray:
    if column not in table.column_names:
        return pa.array([0] * table.num_rows, type=pa.int8())
    return pc.cast(pc.is_valid(table[column]), pa.int8())


def _module_lookup_table(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    if (
        table.num_rows == 0
        or "path" not in table.column_names
        or "module" not in table.column_names
    ):
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=["path", "module"],
        )
    plan = Plan.table(table)
    filters: list[object] = [E.is_valid("path"), E.is_valid("module")]
    if repo is not None and "repo" in table.column_names:
        filters.append(E.or_(E.is_null("repo"), E.field("repo") == E.scalar(repo)))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.or_(E.is_null("commit"), E.field("commit") == E.scalar(commit)))
    plan = plan.filter(E.and_(*filters))
    projection: dict[str, object] = {
        "path": E.cast(E.field("path"), "string"),
        "module": E.cast(E.field("module"), "string"),
    }
    if "repo" in table.column_names:
        projection["repo"] = E.field("repo")
    if "commit" in table.column_names:
        projection["commit"] = E.field("commit")
    plan = plan.project(projection)
    filtered = materialize_plan(plan, use_threads=True)
    if filtered.num_rows == 0:
        return filtered.select(["path", "module"])
    specificity = pc.add(
        _presence_flag(filtered, column="repo"),
        _presence_flag(filtered, column="commit"),
    )
    filtered = filtered.append_column("specificity", specificity)
    deduped = dedupe_keep_first_after_sort(
        filtered,
        key_columns=("path",),
        prefer_columns=("specificity",),
        tie_breakers=(("module", "ascending"),),
    )
    return deduped.select(["path", "module"])


def _symbol_module_edge_counts(
    edge_table: pa.Table,
    module_lookup: pa.Table,
) -> pa.Table:
    if edge_table.num_rows == 0 or module_lookup.num_rows == 0:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.float64()),
            ],
            names=["use_module", "def_module", "weight"],
        )
    if "use_path" not in edge_table.column_names or "def_path" not in edge_table.column_names:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.float64()),
            ],
            names=["use_module", "def_module", "weight"],
        )
    edge_plan = Plan.table(edge_table)
    edge_plan = edge_plan.project(
        {
            "use_path": E.cast(E.field("use_path"), "string"),
            "def_path": E.cast(E.field("def_path"), "string"),
        }
    )
    edge_plan = edge_plan.filter(E.and_(E.is_valid("use_path"), E.is_valid("def_path")))
    module_plan = Plan.table(module_lookup)
    module_plan = module_plan.project(
        {
            "path": E.cast(E.field("path"), "string"),
            "module": E.cast(E.field("module"), "string"),
        }
    )
    module_plan = module_plan.filter(E.and_(E.is_valid("path"), E.is_valid("module")))
    def_join = edge_plan.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["def_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_path"],
            right_output=["module"],
        ),
    )
    def_join = def_join.project({"use_path": E.field("use_path"), "def_module": E.field("module")})
    use_join = def_join.hash_join(
        right=module_plan,
        spec=HashJoinSpec(
            left_keys=["use_path"],
            right_keys=["path"],
            how="inner",
            left_output=["use_path", "def_module"],
            right_output=["module"],
        ),
    )
    use_join = use_join.project(
        {
            "use_module": E.field("module"),
            "def_module": E.field("def_module"),
        }
    )
    use_join = use_join.filter(E.and_(E.is_valid("use_module"), E.is_valid("def_module")))
    use_join = use_join.filter(E.field("use_module") != E.field("def_module"))
    use_join = use_join.aggregate(
        keys=[E.field("use_module"), E.field("def_module")],
        aggregates=[("use_module", "count", None, "weight")],
    )
    use_join = use_join.order_by(
        sort_keys=[("use_module", "ascending"), ("def_module", "ascending")]
    )
    return materialize_plan(use_join, use_threads=True)


def _maybe_to_gpu_graph(store: RxGraphStore, *, use_gpu: bool) -> RxGraphStore:
    """
    No-op for rustworkx-backed execution (CPU-only).

    Parameters
    ----------
    store : RxGraphStore
        Graph store to optionally prepare for GPU execution.
    use_gpu : bool
        Whether GPU execution was requested.

    Returns
    -------
    RxGraphStore
        The original graph store.
    """
    if use_gpu:
        log.debug("GPU backend requested; rustworkx execution is CPU-only.")
    return store


def module_attrs_from_row(
    module: object,
    scc_id: object | None,
    component_size: object | None,
    layer: object | None,
) -> tuple[str, dict[str, int]]:
    """
    Build a normalized node attribute mapping for an import module row.

    Parameters
    ----------
    module :
        Module identifier from the import_modules table.
    scc_id :
        Strongly connected component identifier.
    component_size :
        Size of the SCC.
    layer :
        Condensation DAG layer.

    Returns
    -------
    tuple[str, dict[str, int]]
        Normalized module name and attribute dictionary.
    """
    module_name = str(module)
    attrs: dict[str, int] = {}
    scc_value = as_int(scc_id)
    if scc_value is not None:
        attrs["scc_id"] = scc_value
    comp_size_value = as_int(component_size)
    if comp_size_value is not None:
        attrs["component_size"] = comp_size_value
    layer_value = as_int(layer)
    if layer_value is not None:
        attrs["layer"] = layer_value
    return module_name, attrs


def _apply_node_attrs[NodeId: Hashable](
    store: RxGraphStore,
    attrs_by_node: Mapping[NodeId, Mapping[str, object]],
) -> None:
    for node_id, attrs in attrs_by_node.items():
        if attrs:
            store.set_node_attrs(node_id, attrs)
        else:
            store.ensure_node(node_id)


def _module_attrs_from_reader(
    factory: GraphViewFactory,
    reader: pa.RecordBatchReader,
) -> dict[str, dict[str, int]]:
    attrs_by_module: dict[str, dict[str, int]] = {}
    for module_row in factory.iter_tuples(reader):
        module_name, attrs = module_attrs_from_row(*module_row)
        attrs_by_module[module_name] = attrs
    return attrs_by_module


def _apply_fallback_layers(
    attrs_by_module: dict[str, dict[str, int]],
    fallback_layers: Mapping[str, int],
) -> None:
    for module, layer in fallback_layers.items():
        attrs = attrs_by_module.setdefault(module, {})
        if "layer" not in attrs:
            attrs["layer"] = layer


def _fallback_layer_by_module(table: pa.Table) -> dict[str, int]:
    if (
        table.num_rows == 0
        or "src_module" not in table.column_names
        or "module_layer" not in table.column_names
    ):
        return {}
    plan = Plan.table(table)
    plan = plan.filter(E.and_(E.is_valid("src_module"), E.is_valid("module_layer")))
    plan = plan.project(
        {"src_module": E.field("src_module"), "module_layer": E.field("module_layer")}
    )
    plan = plan.aggregate(
        keys=[E.field("src_module")],
        aggregates=[("module_layer", "max", None, "module_layer_max")],
    )
    plan = plan.order_by(sort_keys=[("src_module", "ascending")])
    grouped = materialize_plan(plan, use_threads=True)
    result: dict[str, int] = {}
    for src_module, layer in _iter_table_tuples(
        grouped,
        columns=("src_module", "module_layer_max"),
    ):
        if src_module is None:
            continue
        layer_value = as_int(layer)
        if layer_value is None:
            continue
        result[str(src_module)] = layer_value
    return result


def load_call_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a call graph store of caller -> callee edges.

    Nodes are GOID integers; parallel edges are aggregated via `weight`.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Directed call graph store with weighted edges and isolated nodes preserved.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.call_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.call_graph_edges",
        columns=("caller_goid_h128", "callee_goid_h128"),
    )
    if edge_reader is None:
        return _empty_graph(directed=True, kind=GraphKind.CALL_GRAPH)

    edge_table = _aggregate_edge_counts(
        reader_to_table(edge_reader),
        src="caller_goid_h128",
        dst="callee_goid_h128",
    )
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("caller_goid_h128", "callee_goid_h128"),
        normalize=normalize_decimal,
    )
    node_attrs: dict[int, dict[str, object]] = {}
    node_reader = factory.load_reader(
        table_key="graph.call_graph_nodes",
        columns=("goid_h128", "kind"),
    )
    if node_reader is not None:
        node_attrs = _call_node_attrs(factory, node_reader)
        if node_attrs:
            node_ids.update(node_attrs.keys())

    store = _edge_table_to_store(
        edge_table,
        spec=_EdgeTableSpec(
            src="caller_goid_h128",
            dst="callee_goid_h128",
            directed=True,
            weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH),
            normalize=normalize_decimal,
        ),
        node_ids=node_ids or None,
    )
    if node_attrs:
        _apply_node_attrs(store, node_attrs)
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_import_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a directed import graph store of module -> module edges.

    Edge weights represent aggregated edge counts when multiple edges exist.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Directed import graph store with weights capturing edge multiplicity.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.import_graph_edges")
    if dataset_root is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.import_graph_edges",
        columns=("src_module", "dst_module", "module_layer"),
    )
    if edge_reader is None:
        return _empty_graph(directed=True, kind=GraphKind.IMPORT_GRAPH)

    edge_table = reader_to_table(edge_reader)
    edge_counts = _aggregate_edge_counts(edge_table, src="src_module", dst="dst_module")
    node_ids = _node_ids_from_table(
        edge_counts,
        columns=("src_module", "dst_module"),
        normalize=_coerce_str,
    )
    fallback_layer_by_module = _fallback_layer_by_module(edge_table)

    module_reader = factory.load_reader(
        table_key="graph.import_modules",
        columns=("module", "scc_id", "component_size", "layer"),
    )
    module_attrs: dict[str, dict[str, int]] = {}
    if module_reader is not None:
        module_attrs = _module_attrs_from_reader(factory, module_reader)
    _apply_fallback_layers(module_attrs, fallback_layer_by_module)
    if module_attrs:
        node_ids.update(module_attrs.keys())

    store = _edge_table_to_store(
        edge_counts,
        spec=_EdgeTableSpec(
            src="src_module",
            dst="dst_module",
            directed=True,
            weight_policy=weight_policy_for_kind(GraphKind.IMPORT_GRAPH),
            normalize=_coerce_str,
        ),
        node_ids=node_ids or None,
    )
    if module_attrs:
        _apply_node_attrs(store, module_attrs)
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def parse_reference_modules(ref_modules: object, allowed_modules: set[str]) -> list[str]:
    """Normalize reference modules input into a filtered list.

    Returns
    -------
    list[str]
        Allowed module names parsed from input.
    """
    modules: list[str] = []
    if isinstance(ref_modules, Mapping):
        ref_modules = ref_modules.get("reference_modules")
    if isinstance(ref_modules, list):
        modules = [str(mod) for mod in ref_modules]
    elif isinstance(ref_modules, str):
        try:
            parsed = json.loads(ref_modules)
            if isinstance(parsed, list):
                modules = [str(mod) for mod in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            modules = []
    if allowed_modules:
        return [module for module in modules if module in allowed_modules]
    return modules


@dataclass
class ConfigGraphStats:
    total_rows: int = 0
    empty_refs: int = 0
    parsed_modules: int = 0
    kept_modules: int = 0
    dropped_modules: int = 0


def _allowed_modules_from_table(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> set[str]:
    if table.num_rows == 0 or "module" not in table.column_names:
        return set()
    filters: list[object] = [E.is_valid("module")]
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))
    plan = Plan.table(table)
    plan = plan.filter(E.and_(*filters))
    plan = plan.project({"module": E.field("module")})
    plan = plan.aggregate(
        keys=[E.field("module")],
        aggregates=[("module", "count", None, "module_count")],
    )
    plan = plan.order_by(sort_keys=[("module", "ascending")])
    filtered = materialize_plan(plan, use_threads=True)
    return {str(row.get("module")) for row in iter_rows(filtered, ("module",)) if row.get("module")}


def _allowed_modules_from_reader(
    factory: GraphViewFactory,
    modules_reader: pa.RecordBatchReader,
) -> set[str]:
    table = reader_to_table(modules_reader)
    return _allowed_modules_from_table(
        table,
        repo=factory.scan_context.repo,
        commit=factory.scan_context.commit,
    )


def _config_bipartite_edges(
    table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
    allowed_modules: set[str],
) -> tuple[
    list[tuple[Hashable, Hashable, float]],
    dict[Hashable, dict[str, object]],
    ConfigGraphStats,
]:
    stats = ConfigGraphStats()
    if table.num_rows == 0 or "key" not in table.column_names or "extras" not in table.column_names:
        return [], {}, stats
    plan = Plan.table(table)
    filters: list[object] = []
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))
    if filters:
        plan = plan.filter(E.and_(*filters))
    plan = plan.project({"key": E.field("key"), "extras": E.field("extras")})
    plan = plan.order_by(sort_keys=[("key", "ascending")])
    filtered = materialize_plan(plan, use_threads=True)
    edges: list[tuple[Hashable, Hashable, float]] = []
    node_attrs: dict[Hashable, dict[str, object]] = {}
    for row in iter_rows(filtered, ("key", "extras")):
        stats.total_rows += 1
        key = row.get("key")
        ref_modules = row.get("extras")
        if key is None or ref_modules is None:
            stats.empty_refs += 1
            continue
        key_node = ("c", str(key))
        node_attrs.setdefault(key_node, {"bipartite": 0})
        raw_modules = parse_reference_modules(ref_modules, set())
        stats.parsed_modules += len(raw_modules)
        filtered_modules = (
            [module for module in raw_modules if module in allowed_modules]
            if allowed_modules
            else raw_modules
        )
        if allowed_modules and raw_modules and not filtered_modules:
            filtered_modules = raw_modules
        stats.kept_modules += len(filtered_modules)
        stats.dropped_modules += len(raw_modules) - len(filtered_modules)
        for module_name in filtered_modules:
            module_node = ("m", module_name)
            node_attrs.setdefault(module_node, {"bipartite": 1})
            edges.append((key_node, module_node, 1.0))
    return edges, node_attrs, stats


def load_config_module_bipartite(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build a bipartite graph store of config keys <-> modules.

    Keys are ("c", key); modules are ("m", module). Edge weight equals one per
    reference occurrence.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected bipartite graph store for configuration references.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "analytics.config_values")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    modules_reader = factory.load_reader(
        table_key="core.modules",
        columns=("module", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if modules_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    allowed_modules = _allowed_modules_from_reader(factory, modules_reader)

    config_reader = factory.load_reader(
        table_key="analytics.config_values",
        columns=("key", "extras", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if config_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)

    config_table = reader_to_table(config_reader)
    edges, node_attrs, stats = _config_bipartite_edges(
        config_table,
        repo=repo,
        commit=commit,
        allowed_modules=allowed_modules,
    )
    if not edges:
        return _empty_graph(directed=False, kind=GraphKind.CONFIG_MODULE_BIPARTITE)
    spec = EdgeBuildSpec(
        directed=False,
        weight_policy=weight_policy_for_kind(GraphKind.CONFIG_MODULE_BIPARTITE),
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_attrs=node_attrs,
        node_hint=len(node_attrs),
        edge_hint=len(edges),
    )
    store = build_store_from_edge_tuples(edges, spec=spec, options=options)
    graph = store.graph
    log.info(
        "Config bipartite built: rows=%d empty_refs=%d allowed_modules=%d "
        "parsed_modules=%d kept_modules=%d dropped_modules=%d graph_nodes=%d edges=%d",
        stats.total_rows,
        stats.empty_refs,
        len(allowed_modules),
        stats.parsed_modules,
        stats.kept_modules,
        stats.dropped_modules,
        graph.num_nodes(),
        graph.num_edges(),
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_module_graph(
    dataset_root: Path | None,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of module-level symbol coupling.

    Edge weights count shared symbol def/use pairs between modules.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    repo : str
        Repository identifier anchoring the view.
    commit : str
        Commit hash anchoring the view.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected graph store where weights reflect shared symbol relations.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    factory = _view_factory(dataset_root, repo=repo, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.symbol_use_edges",
        columns=("def_path", "use_path"),
    )
    if edge_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    module_reader = factory.load_reader(
        table_key="core.modules",
        columns=("path", "module", "repo", "commit"),
        scan_options=GraphViewScanOptions(apply_filter=False),
    )
    if module_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    module_table = reader_to_table(module_reader)
    module_lookup = _module_lookup_table(module_table, repo=repo, commit=commit)
    if module_lookup.num_rows == 0:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_MODULE_GRAPH)
    edge_table = reader_to_table(edge_reader)
    edge_counts = _symbol_module_edge_counts(edge_table, module_lookup)
    node_ids = _node_ids_from_table(
        edge_counts,
        columns=("use_module", "def_module"),
        normalize=_coerce_str,
    )
    store = _edge_table_to_store(
        edge_counts,
        spec=_EdgeTableSpec(
            src="use_module",
            dst="def_module",
            directed=False,
            weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_MODULE_GRAPH),
            normalize=_coerce_str,
            aggregate_edges=True,
        ),
        node_ids=node_ids or None,
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


def load_symbol_function_graph(
    dataset_root: Path | None,
    commit: str,
    *,
    use_gpu: bool = False,
) -> RxGraphStore:
    """
    Build an undirected weighted graph store of function-level symbol coupling (GOIDs).

    Edge weights count shared symbol def/use pairs between functions when available.

    Parameters
    ----------
    dataset_root :
        Root directory for Parquet dataset snapshots.
    commit : str
        Commit hash anchoring the snapshot.
    use_gpu : bool, optional
        Whether to prefer a GPU-backed graph when supported.

    Returns
    -------
    RxGraphStore
        Undirected graph store linking functions by shared symbol usage.
    """
    dataset_root = _ensure_dataset_root(dataset_root, "graph.symbol_use_edges")
    if dataset_root is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)
    factory = _view_factory(dataset_root, repo=None, commit=commit)
    edge_reader = factory.load_reader(
        table_key="graph.symbol_use_edges",
        columns=("def_goid_h128", "use_goid_h128"),
    )
    if edge_reader is None:
        return _empty_graph(directed=False, kind=GraphKind.SYMBOL_FUNCTION_GRAPH)

    edge_table = _aggregate_edge_counts(
        reader_to_table(edge_reader),
        src="use_goid_h128",
        dst="def_goid_h128",
    )
    if edge_table.num_rows:
        plan = Plan.table(edge_table)
        plan = plan.filter(E.field("use_goid_h128") != E.field("def_goid_h128"))
        edge_table = materialize_plan(plan, use_threads=True)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("use_goid_h128", "def_goid_h128"),
        normalize=normalize_decimal,
    )
    store = _edge_table_to_store(
        edge_table,
        spec=_EdgeTableSpec(
            src="use_goid_h128",
            dst="def_goid_h128",
            directed=False,
            weight_policy=weight_policy_for_kind(GraphKind.SYMBOL_FUNCTION_GRAPH),
            normalize=normalize_decimal,
            aggregate_edges=True,
        ),
        node_ids=node_ids or None,
    )
    return _maybe_to_gpu_graph(store, use_gpu=use_gpu)


__all__ = [
    "as_int",
    "load_call_graph",
    "load_config_module_bipartite",
    "load_import_graph",
    "load_symbol_function_graph",
    "load_symbol_module_graph",
    "module_attrs_from_row",
    "normalize_decimal",
    "parse_reference_modules",
]
