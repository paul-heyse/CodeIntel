"""Shared graph builders and edge weight helpers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from codeintel.build.graphs.assembly import table_to_reader
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    BulkEdgeInserter,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata
from codeintel.build.graphs.rx.policies import (
    DEFAULT_NUMERIC_POLICY,
    GraphWeightPolicy,
    weight_policy_for_kind,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.dedupe_ops import DedupeTier
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.core.columnar.iter import iter_array_values, iter_tuples
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.data_models.ids import as_int, normalize_decimal_id
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service

CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"
CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"


def add_weighted_edge(
    store: RxGraphStore,
    source: Hashable,
    target: Hashable,
) -> None:
    """Add or increment a weighted edge in the provided graph."""
    store.add_edge(source, target)


def add_call_graph_edges(
    store: RxGraphStore,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Append call graph edges from row mappings."""
    inserter = BulkEdgeInserter(store=store)
    for row in rows:
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        inserter.add(caller, callee, weight=1.0)
    inserter.flush()


def add_call_graph_nodes(
    store: RxGraphStore,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Append call graph nodes from row mappings."""
    for row in rows:
        node_id = normalize_decimal_id(row.get("goid_h128"))
        if node_id is None:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        store.set_node_attrs(node_id, attrs)


def build_call_graph_from_rows(
    call_graph_edges: Iterable[Mapping[str, object]],
    call_graph_nodes: Iterable[Mapping[str, object]] | None = None,
) -> RxGraphStore:
    """Build a call graph from scoped call graph edge/node rows.

    Returns
    -------
    RxGraphStore
        Directed call graph store populated from the provided rows.
    """
    edges_table = _finalize_rows_table(CALL_GRAPH_EDGES_TABLE_KEY, call_graph_edges)
    nodes_table = (
        _finalize_rows_table(CALL_GRAPH_NODES_TABLE_KEY, call_graph_nodes)
        if call_graph_nodes is not None
        else None
    )
    return build_call_graph_from_tables(edges_table, nodes_table)


def add_import_edges(
    store: RxGraphStore,
    rows: Iterable[Mapping[str, object]],
    *,
    coerce_int: Callable[[object], int | None] = as_int,
) -> dict[str, int]:
    """Append import graph edges and return inferred layer defaults.

    Returns
    -------
    dict[str, int]
        Fallback layer assignments keyed by module name.
    """
    fallback_layer_by_module: dict[str, int] = {}
    inserter = BulkEdgeInserter(store=store)
    for row in rows:
        source_raw = row.get("src_module")
        target_raw = row.get("dst_module")
        if source_raw is None or target_raw is None:
            continue
        source = str(source_raw)
        target = str(target_raw)
        layer = coerce_int(row.get("module_layer"))
        if layer is not None:
            fallback_layer_by_module[source] = layer
        inserter.add(source, target, weight=1.0)
    inserter.flush()
    return fallback_layer_by_module


def add_import_module_rows(
    store: RxGraphStore,
    rows: Iterable[Mapping[str, object]] | None,
    *,
    fallback_layer_by_module: Mapping[str, int],
    coerce_int: Callable[[object], int | None] = as_int,
) -> None:
    """Append import module rows as nodes with attributes."""
    module_rows = list(rows or [])
    if module_rows:
        for row in module_rows:
            module = row.get("module")
            if module is None:
                continue
            module_name = str(module)
            attrs: dict[str, int] = {}
            scc_id = coerce_int(row.get("scc_id"))
            if scc_id is not None:
                attrs["scc_id"] = scc_id
            component_size = coerce_int(row.get("component_size"))
            if component_size is not None:
                attrs["component_size"] = component_size
            layer = coerce_int(row.get("layer"))
            if layer is not None:
                attrs["layer"] = layer
            if "layer" not in attrs and module_name in fallback_layer_by_module:
                attrs["layer"] = fallback_layer_by_module[module_name]
            store.set_node_attrs(module_name, attrs)
        return
    if fallback_layer_by_module:
        for module, layer in fallback_layer_by_module.items():
            store.set_node_attrs(module, {"layer": layer})


def build_import_graph_from_rows(
    import_graph_edges: Iterable[Mapping[str, object]],
    import_modules: Iterable[Mapping[str, object]] | None = None,
    *,
    coerce_int: Callable[[object], int | None] = as_int,
) -> RxGraphStore:
    """Build an import graph from scoped import edges and module rows.

    Returns
    -------
    RxGraphStore
        Directed import graph store populated from the provided rows.
    """
    edges_table = _finalize_rows_table(
        IMPORT_GRAPH_EDGES_TABLE_KEY,
        _coerce_import_edge_rows(import_graph_edges, coerce_int=coerce_int),
    )
    modules_table = (
        _finalize_rows_table(
            IMPORT_MODULES_TABLE_KEY,
            _coerce_import_module_rows(import_modules or (), coerce_int=coerce_int),
        )
        if import_modules is not None
        else None
    )
    return build_import_graph_from_tables(edges_table, modules_table)


def _map_path_to_module(value: object, module_by_path: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    return module_by_path.get(str(value))


def build_symbol_module_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
    *,
    policy: GraphWeightPolicy | None = None,
) -> RxGraphStore:
    """Build an undirected weighted symbol-module graph from use edges.

    Returns
    -------
    RxGraphStore
        Undirected symbol-module graph store populated from the provided rows.
    """
    edges_table = _finalize_rows_table(SYMBOL_USE_EDGES_TABLE_KEY, symbol_use_edges)
    module_table = _module_map_table(module_by_path)
    return build_symbol_module_graph_from_tables(
        edges_table,
        module_table,
        policy=policy,
    )


def build_symbol_function_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    *,
    policy: GraphWeightPolicy | None = None,
) -> RxGraphStore:
    """Build an undirected weighted symbol-function graph from use edges.

    Returns
    -------
    RxGraphStore
        Undirected symbol-function graph store populated from the provided rows.
    """
    edges_table = _finalize_rows_table(SYMBOL_USE_EDGES_TABLE_KEY, symbol_use_edges)
    return build_symbol_function_graph_from_tables(edges_table, policy=policy)


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _determinism_for_table(table_key: str) -> DedupeTier:
    schema = get_schema_service().get_table_schema(table_key)
    if schema is not None:
        policy = schema.finalize_policy
        if policy is not None and policy.dedupe is not None and policy.dedupe.tier is not None:
            return policy.dedupe.tier
    canonical_keys = resolve_canonical_sort_keys(schema)
    if canonical_keys == ():
        return "throughput"
    if canonical_keys:
        return "canonical"
    return "stable_set"


def _ordering_keys_for_table(table_key: str) -> tuple[str, ...] | None:
    schema = get_schema_service().get_table_schema(table_key)
    keys = resolve_canonical_sort_keys(schema)
    if not keys:
        return None
    return tuple(keys)


def _finalize_rows_table(
    table_key: str,
    rows: Iterable[Mapping[str, object]],
) -> pa.Table:
    table, _row_count = table_for_rows(table_key, rows)
    spec = finalize_spec_for_table(
        table_key,
        mode="tolerant",
        determinism=_determinism_for_table(table_key),
    )
    return finalize_table(table, spec=spec).good


def _finalize_table_for_key(
    table_key: str,
    table: pa.Table | None,
) -> pa.Table | None:
    if table is None:
        return None
    spec = finalize_spec_for_table(
        table_key,
        mode="tolerant",
        determinism=_determinism_for_table(table_key),
    )
    return finalize_table(table, spec=spec).good


def _module_map_table(module_by_path: Mapping[str, str]) -> pa.Table:
    if not module_by_path:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()), pa.array([], type=pa.string())],
            names=["path", "module"],
        )
    items = sorted(module_by_path.items())
    paths = [item[0] for item in items]
    modules = [item[1] for item in items]
    return pa.Table.from_arrays(
        [pa.array(paths, type=pa.string()), pa.array(modules, type=pa.string())],
        names=["path", "module"],
    )


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


def _append_snapshot_filters(
    filters: list[object],
    *,
    table: pa.Table,
    repo: str | None,
    commit: str | None,
) -> None:
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))


def _call_graph_edge_table(
    edges_table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    missing = [
        name
        for name in ("caller_goid_h128", "callee_goid_h128")
        if name not in edges_table.column_names
    ]
    if missing:
        msg = f"Missing call graph edge columns: {missing}"
        raise ValueError(msg)
    filters: list[object] = [
        E.is_valid("caller_goid_h128"),
        E.is_valid("callee_goid_h128"),
    ]
    _append_snapshot_filters(filters, table=edges_table, repo=repo, commit=commit)
    plan = build_table_plan(
        table=edges_table,
        options=TablePlanOptions(filter_expr=E.and_(*filters)),
    )
    plan = plan.project(
        {
            "src": E.field("caller_goid_h128"),
            "dst": E.field("callee_goid_h128"),
        }
    )
    plan = plan.aggregate(
        keys=[E.field("src"), E.field("dst")],
        aggregates=[("src", "count", None, "weight")],
    )
    plan = plan.order_by(sort_keys=[("src", "ascending"), ("dst", "ascending")])
    return _plan_to_table(plan)


def _call_graph_node_attrs(
    nodes_table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> dict[int, dict[str, object]]:
    if "goid_h128" not in nodes_table.column_names:
        msg = "Missing call graph node column: goid_h128"
        raise ValueError(msg)
    filters: list[object] = []
    _append_snapshot_filters(filters, table=nodes_table, repo=repo, commit=commit)
    plan = build_table_plan(
        table=nodes_table,
        options=TablePlanOptions(filter_expr=E.and_(*filters)) if filters else None,
    )
    projection: dict[str, object] = {"goid_h128": E.field("goid_h128")}
    if "kind" in nodes_table.column_names:
        projection["kind"] = E.field("kind")
    plan = plan.project(projection)
    plan = plan.order_by(sort_keys=[("goid_h128", "ascending")])
    table = _plan_to_table(plan)
    columns = ["goid_h128"]
    if "kind" in table.column_names:
        columns.append("kind")
    attrs_by_node: dict[int, dict[str, object]] = {}
    for values in iter_tuples(table_to_reader(table), columns=columns):
        node_id = normalize_decimal_id(values[0])
        if node_id is None:
            continue
        attrs: dict[str, object] = {}
        if len(values) > 1 and values[1] is not None:
            attrs["kind"] = str(values[1])
        attrs_by_node[node_id] = attrs
    return attrs_by_node


def _import_graph_edge_table(
    edges_table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    missing = [
        name for name in ("src_module", "dst_module") if name not in edges_table.column_names
    ]
    if missing:
        msg = f"Missing import graph edge columns: {missing}"
        raise ValueError(msg)
    filters: list[object] = [E.is_valid("src_module"), E.is_valid("dst_module")]
    _append_snapshot_filters(filters, table=edges_table, repo=repo, commit=commit)
    plan = build_table_plan(
        table=edges_table,
        options=TablePlanOptions(filter_expr=E.and_(*filters)),
    )
    plan = plan.project({"src": E.field("src_module"), "dst": E.field("dst_module")})
    plan = plan.aggregate(
        keys=[E.field("src"), E.field("dst")],
        aggregates=[("src", "count", None, "weight")],
    )
    plan = plan.order_by(sort_keys=[("src", "ascending"), ("dst", "ascending")])
    return _plan_to_table(plan)


def _import_layer_fallback(
    edges_table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> dict[str, int]:
    if "module_layer" not in edges_table.column_names:
        return {}
    filters: list[object] = [E.is_valid("src_module"), E.is_valid("module_layer")]
    _append_snapshot_filters(filters, table=edges_table, repo=repo, commit=commit)
    plan = build_table_plan(
        table=edges_table,
        options=TablePlanOptions(filter_expr=E.and_(*filters)),
    )
    plan = plan.project(
        {"src_module": E.field("src_module"), "module_layer": E.field("module_layer")}
    )
    plan = plan.aggregate(
        keys=[E.field("src_module")],
        aggregates=[("module_layer", "max", None, "module_layer_max")],
    )
    plan = plan.order_by(sort_keys=[("src_module", "ascending")])
    aggregated = _plan_to_table(plan)
    fallback: dict[str, int] = {}
    for module, layer in iter_tuples(
        table_to_reader(aggregated),
        columns=("src_module", "module_layer_max"),
    ):
        if module is None:
            continue
        layer_value = as_int(layer)
        if layer_value is None:
            continue
        fallback[str(module)] = layer_value
    return fallback


def _import_module_attrs(
    modules_table: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> dict[str, dict[str, int]]:
    if "module" not in modules_table.column_names:
        msg = "Missing import module column: module"
        raise ValueError(msg)
    filters: list[object] = []
    _append_snapshot_filters(filters, table=modules_table, repo=repo, commit=commit)
    plan = build_table_plan(
        table=modules_table,
        options=TablePlanOptions(filter_expr=E.and_(*filters)) if filters else None,
    )
    projection: dict[str, object] = {"module": E.field("module")}
    for column in ("scc_id", "component_size", "layer"):
        if column in modules_table.column_names:
            projection[column] = E.field(column)
    plan = plan.project(projection)
    plan = plan.order_by(sort_keys=[("module", "ascending")])
    table = _plan_to_table(plan)
    columns = list(projection.keys())
    attrs_by_module: dict[str, dict[str, int]] = {}
    for values in iter_tuples(table_to_reader(table), columns=columns):
        module = values[0]
        if module is None:
            continue
        module_name = str(module)
        attrs: dict[str, int] = {}
        for name, value in zip(columns[1:], values[1:], strict=False):
            if value is None:
                continue
            coerced = as_int(value)
            if coerced is None:
                continue
            attrs[name] = coerced
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


def _plan_to_table(plan: Plan) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def _normalize_node_attrs[NodeKey: Hashable](
    node_attrs: Mapping[NodeKey, Mapping[str, object]] | None,
) -> Mapping[Hashable, Mapping[str, object]] | None:
    if not node_attrs:
        return None
    normalized = dict(node_attrs)
    return cast("dict[Hashable, Mapping[str, object]]", normalized)


@dataclass(frozen=True, slots=True)
class _EdgeTableBuildSpec:
    directed: bool
    weight_policy: GraphWeightPolicy
    normalize: Callable[[object], Hashable | None]
    node_ids: Iterable[Hashable] | None
    node_attrs: Mapping[Hashable, Mapping[str, object]] | None


def _build_store_from_edge_table(
    edge_table: pa.Table,
    *,
    spec: _EdgeTableBuildSpec,
) -> RxGraphStore:
    edge_spec = EdgeBuildSpec(
        directed=spec.directed,
        weight_policy=spec.weight_policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=spec.normalize,
        dst_fn=spec.normalize,
    )
    edge_rows = iter_tuples(
        table_to_reader(edge_table),
        columns=("src", "dst", "weight"),
    )
    resolved_node_ids = list(spec.node_ids) if spec.node_ids is not None else None
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=False,
        node_ids=resolved_node_ids,
        node_attrs=spec.node_attrs,
        node_hint=len(resolved_node_ids) if resolved_node_ids is not None else None,
        edge_hint=edge_table.num_rows,
    )
    return build_store_from_edge_tuples(edge_rows, spec=edge_spec, options=options)


def build_call_graph_from_tables(
    call_graph_edges: pa.Table,
    call_graph_nodes: pa.Table | None = None,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> RxGraphStore:
    """Build a call graph from Arrow tables with Acero aggregation.

    Returns
    -------
    RxGraphStore
        Call graph store populated from the provided tables.
    """
    edges = _finalize_table_for_key(CALL_GRAPH_EDGES_TABLE_KEY, call_graph_edges)
    nodes = _finalize_table_for_key(CALL_GRAPH_NODES_TABLE_KEY, call_graph_nodes)
    if edges is None:
        edges = call_graph_edges
    edge_table = _call_graph_edge_table(edges, repo=repo, commit=commit)
    node_attrs: dict[int, dict[str, object]] = {}
    if nodes is not None and nodes.num_rows > 0:
        node_attrs = _call_graph_node_attrs(nodes, repo=repo, commit=commit)
    normalized_attrs = _normalize_node_attrs(node_attrs)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=normalize_decimal_id,
    )
    node_ids.update(node_attrs.keys())
    policy = weight_policy_for_kind(GraphKind.CALL_GRAPH)
    store = _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=True,
            weight_policy=policy,
            normalize=normalize_decimal_id,
            node_ids=node_ids or None,
            node_attrs=normalized_attrs,
        ),
    )
    _apply_graph_metadata(
        store,
        graph_kind=GraphKind.CALL_GRAPH,
        ordering_keys=("caller_goid_h128", "callee_goid_h128"),
        table_key=CALL_GRAPH_EDGES_TABLE_KEY,
    )
    return store


def build_import_graph_from_tables(
    import_graph_edges: pa.Table,
    import_modules: pa.Table | None = None,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> RxGraphStore:
    """Build an import graph from Arrow tables with Acero aggregation.

    Returns
    -------
    RxGraphStore
        Import graph store populated from the provided tables.
    """
    edges = _finalize_table_for_key(IMPORT_GRAPH_EDGES_TABLE_KEY, import_graph_edges)
    modules = _finalize_table_for_key(IMPORT_MODULES_TABLE_KEY, import_modules)
    if edges is None:
        edges = import_graph_edges
    edge_table = _import_graph_edge_table(edges, repo=repo, commit=commit)
    fallback_layers = _import_layer_fallback(edges, repo=repo, commit=commit)
    module_attrs: dict[str, dict[str, int]] = {}
    if modules is not None and modules.num_rows > 0:
        module_attrs = _import_module_attrs(modules, repo=repo, commit=commit)
    _apply_fallback_layers(module_attrs, fallback_layers)
    normalized_attrs = _normalize_node_attrs(module_attrs)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=_coerce_str,
    )
    node_ids.update(module_attrs.keys())
    policy = weight_policy_for_kind(GraphKind.IMPORT_GRAPH)
    store = _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=True,
            weight_policy=policy,
            normalize=_coerce_str,
            node_ids=node_ids or None,
            node_attrs=normalized_attrs,
        ),
    )
    _apply_graph_metadata(
        store,
        graph_kind=GraphKind.IMPORT_GRAPH,
        ordering_keys=("src_module", "dst_module"),
        table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
    )
    return store


def _symbol_module_edge_table(
    symbol_use_edges: pa.Table,
    module_map: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    missing_edges = [
        name for name in ("def_path", "use_path") if name not in symbol_use_edges.column_names
    ]
    if missing_edges:
        msg = f"Missing symbol use columns: {missing_edges}"
        raise ValueError(msg)
    missing_modules = [name for name in ("path", "module") if name not in module_map.column_names]
    if missing_modules:
        msg = f"Missing module map columns: {missing_modules}"
        raise ValueError(msg)
    filters: list[object] = [E.is_valid("def_path"), E.is_valid("use_path")]
    _append_snapshot_filters(filters, table=symbol_use_edges, repo=repo, commit=commit)
    plan = build_table_plan(
        table=symbol_use_edges,
        options=TablePlanOptions(
            filter_expr=E.and_(*filters),
            projection={"def_path": E.field("def_path"), "use_path": E.field("use_path")},
        ),
    )
    module_filters: list[object] = [E.is_valid("path"), E.is_valid("module")]
    _append_snapshot_filters(module_filters, table=module_map, repo=repo, commit=commit)
    module_plan = build_table_plan(
        table=module_map,
        options=TablePlanOptions(
            filter_expr=E.and_(*module_filters),
            projection={"path": E.field("path"), "module": E.field("module")},
            order_by=(("path", "ascending"), ("module", "ascending")),
        ),
    )
    plan = plan.hash_join(
        right=module_plan,
        spec=HashJoinSpec(left_keys=("def_path",), right_keys=("path",), how="inner"),
    )
    plan = plan.project(
        {
            "def_path": E.field("def_path"),
            "use_path": E.field("use_path"),
            "def_module": E.field("module"),
        }
    )
    plan = plan.hash_join(
        right=module_plan,
        spec=HashJoinSpec(left_keys=("use_path",), right_keys=("path",), how="inner"),
    )
    plan = plan.project(
        {
            "use_module": E.field("module"),
            "def_module": E.field("def_module"),
        }
    )
    plan = plan.filter(E.field("use_module") != E.field("def_module"))
    plan = plan.aggregate(
        keys=[E.field("use_module"), E.field("def_module")],
        aggregates=[("use_module", "count", None, "weight")],
    )
    plan = plan.order_by(
        sort_keys=(("use_module", "ascending"), ("def_module", "ascending")),
    )
    return _plan_to_table(plan)


def _symbol_function_edge_table(
    symbol_use_edges: pa.Table,
    *,
    repo: str | None,
    commit: str | None,
) -> pa.Table:
    missing = [
        name
        for name in ("def_goid_h128", "use_goid_h128")
        if name not in symbol_use_edges.column_names
    ]
    if missing:
        msg = f"Missing symbol use columns: {missing}"
        raise ValueError(msg)
    filters: list[object] = [E.is_valid("def_goid_h128"), E.is_valid("use_goid_h128")]
    _append_snapshot_filters(filters, table=symbol_use_edges, repo=repo, commit=commit)
    plan = build_table_plan(
        table=symbol_use_edges,
        options=TablePlanOptions(
            filter_expr=E.and_(*filters),
            projection={
                "src": E.field("use_goid_h128"),
                "dst": E.field("def_goid_h128"),
            },
        ),
    )
    plan = plan.filter(E.field("src") != E.field("dst"))
    plan = plan.aggregate(
        keys=[E.field("src"), E.field("dst")],
        aggregates=[("src", "count", None, "weight")],
    )
    plan = plan.order_by(sort_keys=(("src", "ascending"), ("dst", "ascending")))
    return _plan_to_table(plan)


def build_symbol_module_graph_from_tables(
    symbol_use_edges: pa.Table,
    module_map: pa.Table,
    *,
    policy: GraphWeightPolicy | None = None,
    repo: str | None = None,
    commit: str | None = None,
) -> RxGraphStore:
    """Build an undirected symbol-module graph from Arrow tables.

    Returns
    -------
    RxGraphStore
        Undirected symbol-module graph store populated from the provided tables.
    """
    edges = _finalize_table_for_key(SYMBOL_USE_EDGES_TABLE_KEY, symbol_use_edges)
    if edges is None:
        edges = symbol_use_edges
    edge_table = _symbol_module_edge_table(edges, module_map, repo=repo, commit=commit)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=_coerce_str,
    )
    resolved_policy = policy or weight_policy_for_kind(GraphKind.SYMBOL_MODULE_GRAPH)
    store = _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=False,
            weight_policy=resolved_policy,
            normalize=_coerce_str,
            node_ids=node_ids or None,
            node_attrs=None,
        ),
    )
    _apply_graph_metadata(
        store,
        graph_kind=GraphKind.SYMBOL_MODULE_GRAPH,
        ordering_keys=("use_module", "def_module"),
        table_key=SYMBOL_USE_EDGES_TABLE_KEY,
    )
    return store


def build_symbol_function_graph_from_tables(
    symbol_use_edges: pa.Table,
    *,
    policy: GraphWeightPolicy | None = None,
    repo: str | None = None,
    commit: str | None = None,
) -> RxGraphStore:
    """Build an undirected symbol-function graph from Arrow tables.

    Returns
    -------
    RxGraphStore
        Undirected symbol-function graph store populated from the provided tables.
    """
    edges = _finalize_table_for_key(SYMBOL_USE_EDGES_TABLE_KEY, symbol_use_edges)
    if edges is None:
        edges = symbol_use_edges
    edge_table = _symbol_function_edge_table(edges, repo=repo, commit=commit)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=normalize_decimal_id,
    )
    resolved_policy = policy or weight_policy_for_kind(GraphKind.SYMBOL_FUNCTION_GRAPH)
    store = _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=False,
            weight_policy=resolved_policy,
            normalize=normalize_decimal_id,
            node_ids=node_ids or None,
            node_attrs=None,
        ),
    )
    _apply_graph_metadata(
        store,
        graph_kind=GraphKind.SYMBOL_FUNCTION_GRAPH,
        ordering_keys=("use_goid_h128", "def_goid_h128"),
        table_key=SYMBOL_USE_EDGES_TABLE_KEY,
    )
    return store


def _graph_kind_name(kind: GraphKind) -> str:
    raw = getattr(kind, "name", None)
    if isinstance(raw, str):
        return raw
    return str(kind)


def _apply_graph_metadata(
    store: RxGraphStore,
    *,
    graph_kind: GraphKind,
    ordering_keys: tuple[str, ...] | None,
    table_key: str | None = None,
) -> None:
    resolved_ordering_keys = ordering_keys
    if resolved_ordering_keys is None and table_key is not None:
        resolved_ordering_keys = _ordering_keys_for_table(table_key)
    determinism_tier = _determinism_for_table(table_key) if table_key is not None else None
    if determinism_tier is None:
        metadata = GraphMetadata(
            weight_policy=store.weight_policy.name,
            graph_kind=_graph_kind_name(graph_kind),
            ordering_keys=resolved_ordering_keys,
        )
    else:
        metadata = GraphMetadata(
            weight_policy=store.weight_policy.name,
            graph_kind=_graph_kind_name(graph_kind),
            ordering_keys=resolved_ordering_keys,
            determinism_tier=determinism_tier,
        )
    apply_graph_metadata(store.graph, metadata)


def _coerce_int_fields(
    row: Mapping[str, object],
    *,
    fields: Sequence[str],
    coerce_int: Callable[[object], int | None],
) -> Mapping[str, object]:
    updated: dict[str, int] = {}
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        coerced = coerce_int(value)
        if coerced is None or coerced == value:
            continue
        updated[field] = coerced
    if not updated:
        return row
    normalized = dict(row)
    normalized.update(updated)
    return normalized


def _coerce_import_edge_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    coerce_int: Callable[[object], int | None],
) -> Iterable[Mapping[str, object]]:
    fields = ("src_fan_out", "dst_fan_in", "cycle_group", "module_layer")
    for row in rows:
        yield _coerce_int_fields(row, fields=fields, coerce_int=coerce_int)


def _coerce_import_module_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    coerce_int: Callable[[object], int | None],
) -> Iterable[Mapping[str, object]]:
    fields = ("scc_id", "component_size", "layer", "cycle_group")
    for row in rows:
        yield _coerce_int_fields(row, fields=fields, coerce_int=coerce_int)


def _collect_import_edge_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    coerce_int: Callable[[object], int | None],
) -> tuple[list[tuple[str, str, float]], set[str], dict[str, int]]:
    edge_rows: list[tuple[str, str, float]] = []
    node_ids: set[str] = set()
    fallback_layer_by_module: dict[str, int] = {}
    for row in rows:
        source_raw = row.get("src_module")
        target_raw = row.get("dst_module")
        if source_raw is None or target_raw is None:
            continue
        source = str(source_raw)
        target = str(target_raw)
        node_ids.update((source, target))
        layer = coerce_int(row.get("module_layer"))
        if layer is not None:
            fallback_layer_by_module[source] = layer
        edge_rows.append((source, target, 1.0))
    return edge_rows, node_ids, fallback_layer_by_module


def _collect_import_module_attrs(
    rows: Iterable[Mapping[str, object]] | None,
    *,
    coerce_int: Callable[[object], int | None],
) -> dict[str, dict[str, int]]:
    if rows is None:
        return {}
    module_attrs: dict[str, dict[str, int]] = {}
    for row in rows:
        module = row.get("module")
        if module is None:
            continue
        module_name = str(module)
        attrs: dict[str, int] = {}
        scc_id = coerce_int(row.get("scc_id"))
        if scc_id is not None:
            attrs["scc_id"] = scc_id
        component_size = coerce_int(row.get("component_size"))
        if component_size is not None:
            attrs["component_size"] = component_size
        layer = coerce_int(row.get("layer"))
        if layer is not None:
            attrs["layer"] = layer
        module_attrs[module_name] = attrs
    return module_attrs


def build_symbol_module_edges(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate symbol use edges to module-level adjacency.

    Returns
    -------
    tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
        Modules, inbound adjacency, and outbound adjacency extracted from use edges.
    """
    edges_table = _finalize_rows_table(SYMBOL_USE_EDGES_TABLE_KEY, symbol_use_edges)
    edge_table = _symbol_module_edge_table(
        edges_table,
        _module_map_table(module_by_path),
        repo=None,
        commit=None,
    )
    modules: set[str] = set()
    inbound: dict[str, set[str]] = {}
    outbound: dict[str, set[str]] = {}
    for src, dst, _weight in iter_tuples(
        table_to_reader(edge_table),
        columns=("src", "dst", "weight"),
    ):
        if src is None or dst is None:
            continue
        src_module = str(src)
        dst_module = str(dst)
        modules.update((src_module, dst_module))
        inbound.setdefault(dst_module, set()).add(src_module)
        outbound.setdefault(src_module, set()).add(dst_module)
    return modules, inbound, outbound


__all__ = [
    "add_call_graph_edges",
    "add_call_graph_nodes",
    "add_import_edges",
    "add_import_module_rows",
    "add_weighted_edge",
    "build_call_graph_from_rows",
    "build_call_graph_from_tables",
    "build_import_graph_from_rows",
    "build_import_graph_from_tables",
    "build_symbol_function_graph",
    "build_symbol_function_graph_from_tables",
    "build_symbol_module_edges",
    "build_symbol_module_graph",
    "build_symbol_module_graph_from_tables",
]
