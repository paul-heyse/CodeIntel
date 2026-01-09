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
from codeintel.build.graphs.rx.policies import (
    DEFAULT_NUMERIC_POLICY,
    GraphWeightPolicy,
    weight_policy_for_kind,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.iter import iter_array_values, iter_tuples
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import materialize_plan
from codeintel.core.data_models.ids import as_int, normalize_decimal_id


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
    policy = weight_policy_for_kind(GraphKind.CALL_GRAPH)
    edge_rows: list[tuple[int, int, float]] = []
    node_ids: set[int] = set()
    for row in call_graph_edges:
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        node_ids.update((caller, callee))
        edge_rows.append((caller, callee, 1.0))
    node_attrs: dict[int, dict[str, object]] = {}
    if call_graph_nodes is not None:
        for row in call_graph_nodes:
            node_id = normalize_decimal_id(row.get("goid_h128"))
            if node_id is None:
                continue
            attrs: dict[str, object] = {}
            kind = row.get("kind")
            if kind is not None:
                attrs["kind"] = str(kind)
            node_attrs[node_id] = attrs
            node_ids.add(node_id)
    if not edge_rows and not node_ids:
        return RxGraphStore.directed(weight_policy=policy)
    if edge_rows:
        edge_rows.sort()
    normalized_attrs = _normalize_node_attrs(node_attrs)
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=normalize_decimal_id,
        dst_fn=normalize_decimal_id,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        node_ids=node_ids or None,
        node_attrs=normalized_attrs,
        node_hint=len(node_ids) if node_ids else None,
        edge_hint=len(edge_rows),
    )
    return build_store_from_edge_tuples(edge_rows, spec=spec, options=options)


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
    policy = weight_policy_for_kind(GraphKind.IMPORT_GRAPH)
    edge_rows, node_ids, fallback_layer_by_module = _collect_import_edge_rows(
        import_graph_edges,
        coerce_int=coerce_int,
    )
    module_attrs = _collect_import_module_attrs(import_modules, coerce_int=coerce_int)
    _apply_fallback_layers(module_attrs, fallback_layer_by_module)
    if module_attrs:
        node_ids.update(module_attrs.keys())
    if not edge_rows and not node_ids:
        return RxGraphStore.directed(weight_policy=policy)
    if edge_rows:
        edge_rows.sort()
    normalized_attrs = _normalize_node_attrs(module_attrs)
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=_coerce_str,
        dst_fn=_coerce_str,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        node_ids=node_ids or None,
        node_attrs=normalized_attrs,
        node_hint=len(node_ids) if node_ids else None,
        edge_hint=len(edge_rows),
    )
    return build_store_from_edge_tuples(edge_rows, spec=spec, options=options)


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
    resolved_policy = policy or weight_policy_for_kind(GraphKind.SYMBOL_MODULE_GRAPH)
    edge_rows: list[tuple[str, str, float]] = []
    node_ids: set[str] = set()
    for record in symbol_use_edges:
        def_module = _map_path_to_module(record.get("def_path"), module_by_path)
        use_module = _map_path_to_module(record.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        node_ids.update((use_module, def_module))
        edge_rows.append((use_module, def_module, 1.0))
    if not edge_rows:
        return RxGraphStore.undirected(weight_policy=resolved_policy)
    edge_rows.sort()
    spec = EdgeBuildSpec(
        directed=False,
        weight_policy=resolved_policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=_coerce_str,
        dst_fn=_coerce_str,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        node_ids=node_ids,
        node_hint=len(node_ids),
        edge_hint=len(edge_rows),
    )
    return build_store_from_edge_tuples(edge_rows, spec=spec, options=options)


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
    resolved_policy = policy or weight_policy_for_kind(GraphKind.SYMBOL_FUNCTION_GRAPH)
    edge_rows: list[tuple[int, int, float]] = []
    node_ids: set[int] = set()
    for record in symbol_use_edges:
        def_goid = normalize_decimal_id(record.get("def_goid_h128"))
        use_goid = normalize_decimal_id(record.get("use_goid_h128"))
        if def_goid is None or use_goid is None:
            continue
        if def_goid == use_goid:
            continue
        node_ids.update((use_goid, def_goid))
        edge_rows.append((use_goid, def_goid, 1.0))
    if not edge_rows:
        return RxGraphStore.undirected(weight_policy=resolved_policy)
    edge_rows.sort()
    spec = EdgeBuildSpec(
        directed=False,
        weight_policy=resolved_policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
        src_fn=normalize_decimal_id,
        dst_fn=normalize_decimal_id,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        node_ids=node_ids,
        node_hint=len(node_ids),
        edge_hint=len(edge_rows),
    )
    return build_store_from_edge_tuples(edge_rows, spec=spec, options=options)


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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
    return materialize_plan(plan, use_threads=True)


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
    table = materialize_plan(plan, use_threads=True)
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
    return materialize_plan(plan, use_threads=True)


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
    aggregated = materialize_plan(plan, use_threads=True)
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
    table = materialize_plan(plan, use_threads=True)
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
    edge_table = _call_graph_edge_table(call_graph_edges, repo=repo, commit=commit)
    node_attrs: dict[int, dict[str, object]] = {}
    if call_graph_nodes is not None and call_graph_nodes.num_rows > 0:
        node_attrs = _call_graph_node_attrs(call_graph_nodes, repo=repo, commit=commit)
    normalized_attrs = _normalize_node_attrs(node_attrs)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=normalize_decimal_id,
    )
    node_ids.update(node_attrs.keys())
    policy = weight_policy_for_kind(GraphKind.CALL_GRAPH)
    return _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=True,
            weight_policy=policy,
            normalize=normalize_decimal_id,
            node_ids=node_ids or None,
            node_attrs=normalized_attrs,
        ),
    )


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
    edge_table = _import_graph_edge_table(import_graph_edges, repo=repo, commit=commit)
    fallback_layers = _import_layer_fallback(import_graph_edges, repo=repo, commit=commit)
    module_attrs: dict[str, dict[str, int]] = {}
    if import_modules is not None and import_modules.num_rows > 0:
        module_attrs = _import_module_attrs(import_modules, repo=repo, commit=commit)
    _apply_fallback_layers(module_attrs, fallback_layers)
    normalized_attrs = _normalize_node_attrs(module_attrs)
    node_ids = _node_ids_from_table(
        edge_table,
        columns=("src", "dst"),
        normalize=_coerce_str,
    )
    node_ids.update(module_attrs.keys())
    policy = weight_policy_for_kind(GraphKind.IMPORT_GRAPH)
    return _build_store_from_edge_table(
        edge_table,
        spec=_EdgeTableBuildSpec(
            directed=True,
            weight_policy=policy,
            normalize=_coerce_str,
            node_ids=node_ids or None,
            node_attrs=normalized_attrs,
        ),
    )


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
    modules: set[str] = set()
    inbound: dict[str, set[str]] = {}
    outbound: dict[str, set[str]] = {}
    for record in symbol_use_edges:
        def_module = _map_path_to_module(record.get("def_path"), module_by_path)
        use_module = _map_path_to_module(record.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        modules.update((use_module, def_module))
        inbound.setdefault(def_module, set()).add(use_module)
        outbound.setdefault(use_module, set()).add(def_module)
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
    "build_symbol_module_edges",
    "build_symbol_module_graph",
]
