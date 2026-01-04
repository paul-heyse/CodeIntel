"""Graph metrics analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import pyarrow as pa
from hamilton.function_modifiers import cache

from codeintel.build.analytics.graphs.config_graph_metrics import (
    build_config_module_bipartite,
)
from codeintel.build.analytics.graphs.graph_metrics import (
    ComponentMeta,
    GraphMetricFilters,
    GraphMetricsInputs,
    GraphMetricsRows,
    SymbolModuleEdges,
    build_graph_metric_filters_from_sets,
    build_graph_metrics_rows,
)
from codeintel.build.analytics.graphs.graph_metrics_ext import (
    build_graph_metrics_functions_ext_rows,
)
from codeintel.build.analytics.graphs.graph_stats import (
    GraphStatsInputs,
    build_graph_stats_rows,
)
from codeintel.build.analytics.graphs.module_graph_metrics_ext import (
    build_graph_metrics_modules_ext_rows,
)
from codeintel.build.analytics.graphs.symbol_graph_metrics import (
    build_symbol_graph_metrics_function_rows,
    build_symbol_graph_metrics_module_rows,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.compute_masks import equal_mask
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_int

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GRAPH_METRICS_TARGET_NAME = "graph_metrics"
GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.graph_metrics_functions"
GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.graph_metrics_modules"
GRAPH_METRICS_COLLECT_GROUP = "graph_metrics_core"
GRAPH_METRICS_FUNCTIONS_CONTRACT = TableContractSpec(
    table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    domain="analytics",
    target=GRAPH_METRICS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_metrics_functions__base",
)
GRAPH_METRICS_MODULES_CONTRACT = TableContractSpec(
    table_key=GRAPH_METRICS_MODULES_TABLE_KEY,
    domain="analytics",
    target=GRAPH_METRICS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_metrics_modules__base",
)

GRAPH_METRICS_EXT_TARGET_NAME = "graph_metrics_ext"
GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY = "analytics.graph_metrics_functions_ext"
GRAPH_METRICS_MODULES_EXT_TABLE_KEY = "analytics.graph_metrics_modules_ext"
GRAPH_METRICS_FUNCTIONS_EXT_CONTRACT = TableContractSpec(
    table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_metrics_functions_ext__base",
)
GRAPH_METRICS_MODULES_EXT_CONTRACT = TableContractSpec(
    table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_metrics_modules_ext__base",
)

SYMBOL_GRAPH_METRICS_TARGET_NAME = "symbol_graph_metrics"
SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY = "analytics.symbol_graph_metrics_functions"
SYMBOL_GRAPH_MODULES_TABLE_KEY = "analytics.symbol_graph_metrics_modules"
SYMBOL_GRAPH_FUNCTIONS_CONTRACT = TableContractSpec(
    table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY,
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="symbol_graph_metrics_functions__base",
)
SYMBOL_GRAPH_MODULES_CONTRACT = TableContractSpec(
    table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY,
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="symbol_graph_metrics_modules__base",
)

GRAPH_STATS_TARGET_NAME = "graph_stats"
GRAPH_STATS_TABLE_KEY = "analytics.graph_stats"
GRAPH_STATS_CONTRACT = TableContractSpec(
    table_key=GRAPH_STATS_TABLE_KEY,
    domain="analytics",
    target=GRAPH_STATS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_stats__base",
)


@dataclass(frozen=True)
class GraphMetricInputs:
    """Shared graph metric inputs derived from DAG sources."""

    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    symbol_module_edges: SymbolModuleEdges
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph
    module_names: set[str]
    function_goids: set[int]
    filters: GraphMetricFilters
    component_meta: ComponentMeta | None
    runtime_options: GraphRuntimeOptions


@dataclass(frozen=True)
class GraphMetricCoreFrames:
    """Core graph inputs sourced from DAG tables."""

    goids: InferableTabularInput
    modules: InferableTabularInput
    call_graph_edges: InferableTabularInput
    call_graph_nodes: InferableTabularInput


@dataclass(frozen=True)
class GraphMetricSupportFrames:
    """Secondary graph inputs sourced from DAG tables."""

    import_graph_edges: InferableTabularInput
    import_modules: InferableTabularInput
    symbol_use_edges: InferableTabularInput
    subsystem_modules: InferableTabularInput


_FUNCTION_KINDS: frozenset[str] = frozenset({"function", "method"})


def _graph_runtime_options(env: BuildEnv) -> GraphRuntimeOptions:
    if env.execution_context is None:
        return GraphRuntimeOptions(snapshot=env.snapshot)
    return GraphRuntimeOptions(
        snapshot=env.snapshot,
        backend=env.execution_context.graph_backend,
        features=env.execution_context.graph_features,
    )


def _collect_rows(
    value: InferableTabularInput,
    columns: tuple[str, ...],
    *,
    repo: str | None,
    commit: str | None,
) -> list[dict[str, object]]:
    table = tabular_to_arrow_table(value)
    if repo is not None and "repo" in table.column_names:
        table = table.filter(equal_mask(table["repo"], pa.scalar(repo)))
    if commit is not None and "commit" in table.column_names:
        table = table.filter(equal_mask(table["commit"], pa.scalar(commit)))
    if columns:
        missing = [col for col in columns if col not in table.column_names]
        if missing:
            msg = f"Missing columns in graph_metrics inputs: {missing}"
            raise ValueError(msg)
        table = table.select(list(columns))
    if table.num_rows == 0:
        return []
    return table.to_pylist()


def _matches_optional_scope_value(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value == expected
    return str(value).strip() == expected


def _filter_rows_by_snapshot(
    rows: list[dict[str, object]],
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        repo_value = row.get("repo")
        commit_value = row.get("commit")
        if repo_value is not None and not _matches_optional_scope_value(repo_value, repo):
            continue
        if commit_value is not None and not _matches_optional_scope_value(commit_value, commit):
            continue
        filtered.append(row)
    return filtered


def _allowed_modules_from_rows(
    rows: list[dict[str, object]],
    *,
    repo: str,
    commit: str,
) -> set[str]:
    if not rows:
        return set()
    filtered = _filter_rows_by_snapshot(rows, repo=repo, commit=commit)
    if not filtered:
        return set()
    return {
        str(row.get("module"))
        for row in filtered
        if row.get("module") is not None
    }


def _load_module_inputs(
    env: BuildEnv, table: InferableTabularInput
) -> tuple[dict[str, str], set[str]]:
    modules_rows = _collect_rows(
        table,
        ("module", "path", "repo", "commit"),
        repo=None,
        commit=None,
    )
    return _module_inputs_from_rows(
        modules_rows,
        repo=env.repo,
        commit=env.commit,
    )


def _load_function_goids(env: BuildEnv, table: InferableTabularInput) -> set[int]:
    goid_rows = _collect_rows(
        table,
        ("goid_h128", "kind"),
        repo=env.repo,
        commit=env.commit,
    )
    return _function_goids_from_rows(goid_rows)


def _load_subsystem_ids(env: BuildEnv, table: InferableTabularInput) -> set[str]:
    subsystem_rows = _collect_rows(
        table,
        ("subsystem_id",),
        repo=env.repo,
        commit=env.commit,
    )
    return _subsystem_ids_from_rows(subsystem_rows)


def _load_call_graph(
    env: BuildEnv,
    edges: InferableTabularInput,
    nodes: InferableTabularInput,
) -> nx.DiGraph:
    call_edge_rows = _collect_rows(
        edges,
        ("caller_goid_h128", "callee_goid_h128"),
        repo=env.repo,
        commit=env.commit,
    )
    call_node_rows = _collect_rows(
        nodes,
        ("goid_h128", "kind"),
        repo=None,
        commit=None,
    )
    return _call_graph_from_rows(call_edge_rows, call_node_rows)


def _load_import_graph(
    env: BuildEnv,
    edges: InferableTabularInput,
    modules: InferableTabularInput,
) -> tuple[nx.DiGraph, ComponentMeta | None]:
    import_edge_rows = _collect_rows(
        edges,
        ("src_module", "dst_module", "module_layer"),
        repo=env.repo,
        commit=env.commit,
    )
    import_module_rows = _collect_rows(
        modules,
        ("module", "scc_id", "component_size", "layer"),
        repo=env.repo,
        commit=env.commit,
    )
    return _import_graph_from_rows(import_edge_rows, import_module_rows)


def _call_graph_from_rows(
    edges: list[dict[str, object]],
    nodes: list[dict[str, object]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    _add_call_graph_edges(graph, edges)
    _add_call_graph_nodes(graph, nodes)
    return graph


def _increment_edge_weight(attrs: Mapping[str, object], *, ctx: str) -> int:
    weight = coerce_optional_int(attrs.get("weight"), ctx=ctx)
    return (weight if weight is not None else 0) + 1


def _add_call_graph_edges(graph: nx.DiGraph, edges: list[dict[str, object]]) -> None:
    if not edges:
        return
    for row in edges:
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        if graph.has_edge(caller, callee):
            attrs = graph[caller][callee]
            attrs["weight"] = _increment_edge_weight(attrs, ctx="call_graph_edge_weight")
        else:
            graph.add_edge(caller, callee, weight=1)


def _add_call_graph_nodes(graph: nx.DiGraph, nodes: list[dict[str, object]]) -> None:
    if not nodes:
        return
    for row in nodes:
        node_id = normalize_decimal_id(row.get("goid_h128"))
        if node_id is None or node_id in graph:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(node_id, **attrs)


def _import_graph_from_rows(
    edges: list[dict[str, object]],
    modules: list[dict[str, object]],
) -> tuple[nx.DiGraph, ComponentMeta | None]:
    graph = nx.DiGraph()
    fallback_layer_by_module: dict[str, int] = {}
    _add_import_edges(graph, edges, fallback_layer_by_module)
    component_meta = _component_meta_from_import_rows(modules)
    _apply_import_module_frame(graph, modules, fallback_layer_by_module)
    return graph, component_meta


def _add_import_edges(
    graph: nx.DiGraph,
    edges: list[dict[str, object]],
    fallback_layer_by_module: dict[str, int],
) -> None:
    if not edges:
        return
    for row in edges:
        source_raw = row.get("src_module")
        target_raw = row.get("dst_module")
        if source_raw is None or target_raw is None:
            continue
        source = str(source_raw)
        target = str(target_raw)
        layer = coerce_optional_int(row.get("module_layer"), ctx="module_layer")
        if layer is not None:
            fallback_layer_by_module[source] = layer
        if graph.has_edge(source, target):
            attrs = graph[source][target]
            attrs["weight"] = _increment_edge_weight(attrs, ctx="import_graph_edge_weight")
        else:
            graph.add_edge(source, target, weight=1)


def _component_meta_from_import_rows(
    rows: list[dict[str, object]],
) -> ComponentMeta | None:
    if not rows:
        return None
    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    found = False
    for row in rows:
        module = row.get("module")
        if module is None:
            continue
        found = True
        module_name = str(module)
        scc_value = coerce_optional_int(row.get("scc_id"), ctx="scc_id")
        size_value = coerce_optional_int(row.get("component_size"), ctx="component_size")
        layer_value = coerce_optional_int(row.get("layer"), ctx="layer")
        comp_id[module_name] = scc_value if scc_value is not None else -1
        in_cycle[module_name] = bool(size_value and size_value > 1)
        layer_by_module[module_name] = layer_value if layer_value is not None else 0
    if not found:
        return None
    return {
        "component_id": comp_id,
        "in_cycle": in_cycle,
        "layer": layer_by_module,
    }


def _apply_import_module_frame(
    graph: nx.DiGraph,
    rows: list[dict[str, object]],
    fallback_layer_by_module: Mapping[str, int],
) -> None:
    if not rows:
        return
    for row in rows:
        module = row.get("module")
        if module is None:
            continue
        module_name = str(module)
        attrs: dict[str, object] = {}
        scc_value = coerce_optional_int(row.get("scc_id"), ctx="scc_id")
        size_value = coerce_optional_int(row.get("component_size"), ctx="component_size")
        layer_value = coerce_optional_int(row.get("layer"), ctx="layer")
        attrs["scc_id"] = scc_value
        attrs["component_size"] = size_value
        attrs["layer"] = layer_value
        if attrs.get("layer") is None and module_name in fallback_layer_by_module:
            attrs["layer"] = fallback_layer_by_module[module_name]
        graph.add_node(module_name, **{k: v for k, v in attrs.items() if v is not None})


def _map_path_to_module(value: object, module_by_path: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    return module_by_path.get(str(value))


def _symbol_module_edges_from_rows(
    rows: list[dict[str, object]],
    module_by_path: Mapping[str, str],
) -> SymbolModuleEdges:
    if not rows:
        return set(), {}, {}
    modules: set[str] = set()
    inbound: dict[str, set[str]] = {}
    outbound: dict[str, set[str]] = {}
    for row in rows:
        def_module = _map_path_to_module(row.get("def_path"), module_by_path)
        use_module = _map_path_to_module(row.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        modules.update({def_module, use_module})
        inbound.setdefault(def_module, set()).add(use_module)
        outbound.setdefault(use_module, set()).add(def_module)
    return modules, inbound, outbound


def _symbol_module_graph_from_rows(
    rows: list[dict[str, object]],
    module_by_path: Mapping[str, str],
) -> nx.Graph:
    graph = nx.Graph()
    if not rows:
        return graph
    edge_weights: dict[tuple[str, str], int] = {}
    for row in rows:
        def_module = _map_path_to_module(row.get("def_path"), module_by_path)
        use_module = _map_path_to_module(row.get("use_path"), module_by_path)
        if def_module is None or use_module is None:
            continue
        if def_module == use_module:
            continue
        key = (use_module, def_module)
        edge_weights[key] = edge_weights.get(key, 0) + 1
    for (use_module, def_module), weight in edge_weights.items():
        graph.add_edge(use_module, def_module, weight=weight)
    return graph


def _symbol_function_graph_from_rows(
    rows: list[dict[str, object]],
) -> nx.Graph:
    graph = nx.Graph()
    if not rows:
        return graph
    edge_weights: dict[tuple[int, int], int] = {}
    for row in rows:
        def_goid = normalize_decimal_id(row.get("def_goid_h128"))
        use_goid = normalize_decimal_id(row.get("use_goid_h128"))
        if def_goid is None or use_goid is None:
            continue
        if def_goid == use_goid:
            continue
        key = (use_goid, def_goid)
        edge_weights[key] = edge_weights.get(key, 0) + 1
    for (use_goid, def_goid), weight in edge_weights.items():
        graph.add_edge(use_goid, def_goid, weight=weight)
    return graph


def _load_symbol_graphs(
    module_by_path: Mapping[str, str],
    table: InferableTabularInput,
) -> tuple[SymbolModuleEdges, nx.Graph, nx.Graph]:
    symbol_rows = _collect_rows(
        table,
        ("def_path", "use_path", "def_goid_h128", "use_goid_h128"),
        repo=None,
        commit=None,
    )
    symbol_module_edges = _symbol_module_edges_from_rows(symbol_rows, module_by_path)
    symbol_module_graph = _symbol_module_graph_from_rows(symbol_rows, module_by_path)
    symbol_function_graph = _symbol_function_graph_from_rows(symbol_rows)
    return symbol_module_edges, symbol_module_graph, symbol_function_graph


def _module_inputs_from_rows(
    rows: list[dict[str, object]],
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, str], set[str]]:
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    if not rows:
        return module_by_path, module_names
    filtered = _filter_rows_by_snapshot(rows, repo=repo, commit=commit)
    if not filtered:
        return module_by_path, module_names
    for row in filtered:
        module = row.get("module")
        if module is not None:
            module_names.add(str(module))
        path = row.get("path")
        if path is None or module is None:
            continue
        module_by_path[str(path)] = str(module)
    return module_by_path, module_names


def _function_goids_from_rows(rows: list[dict[str, object]]) -> set[int]:
    function_goids: set[int] = set()
    if not rows:
        return function_goids
    for row in rows:
        kind = row.get("kind")
        if kind is not None and kind not in _FUNCTION_KINDS:
            continue
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is not None:
            function_goids.add(goid)
    return function_goids


def _subsystem_ids_from_rows(rows: list[dict[str, object]]) -> set[str]:
    subsystem_ids: set[str] = set()
    if not rows:
        return subsystem_ids
    for row in rows:
        subsystem_id = row.get("subsystem_id")
        if subsystem_id is not None:
            subsystem_ids.add(str(subsystem_id))
    return subsystem_ids


def graph_metric_core_frames(
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__call_graph_nodes: InferableTabularInput,
) -> GraphMetricCoreFrames:
    """Bundle core graph tables for metric computation.

    Returns
    -------
    GraphMetricCoreFrames
        Core graph inputs for metric computation.
    """
    return GraphMetricCoreFrames(
        goids=q__core__goids,
        modules=q__core__modules,
        call_graph_edges=q__graph__call_graph_edges,
        call_graph_nodes=q__graph__call_graph_nodes,
    )


def graph_metric_support_frames(
    q__graph__import_graph_edges: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
    q__graph__symbol_use_edges: InferableTabularInput,
    q__analytics__subsystem_modules: InferableTabularInput,
) -> GraphMetricSupportFrames:
    """Bundle support graph tables for metric computation.

    Returns
    -------
    GraphMetricSupportFrames
        Support graph inputs for metric computation.
    """
    return GraphMetricSupportFrames(
        import_graph_edges=q__graph__import_graph_edges,
        import_modules=q__graph__import_modules,
        symbol_use_edges=q__graph__symbol_use_edges,
        subsystem_modules=q__analytics__subsystem_modules,
    )


def graph_metric_inputs(
    env: BuildEnv,
    graph_metric_core_frames: GraphMetricCoreFrames,
    graph_metric_support_frames: GraphMetricSupportFrames,
) -> GraphMetricInputs:
    """Assemble shared graph metric inputs from DAG-provided tables.

    Returns
    -------
    GraphMetricInputs
        Structured graph metric inputs for downstream nodes.
    """
    runtime_options = _graph_runtime_options(env)
    module_by_path, module_names = _load_module_inputs(env, graph_metric_core_frames.modules)
    function_goids = _load_function_goids(env, graph_metric_core_frames.goids)
    subsystem_ids = _load_subsystem_ids(env, graph_metric_support_frames.subsystem_modules)
    filters = build_graph_metric_filters_from_sets(
        function_goids=function_goids,
        modules=module_names,
        subsystems=subsystem_ids,
    )
    call_graph = _load_call_graph(
        env,
        graph_metric_core_frames.call_graph_edges,
        graph_metric_core_frames.call_graph_nodes,
    )
    import_graph, component_meta = _load_import_graph(
        env,
        graph_metric_support_frames.import_graph_edges,
        graph_metric_support_frames.import_modules,
    )
    symbol_module_edges, symbol_module_graph, symbol_function_graph = _load_symbol_graphs(
        module_by_path,
        graph_metric_support_frames.symbol_use_edges,
    )
    return GraphMetricInputs(
        call_graph=call_graph,
        import_graph=import_graph,
        symbol_module_edges=symbol_module_edges,
        symbol_module_graph=symbol_module_graph,
        symbol_function_graph=symbol_function_graph,
        module_names=module_names,
        function_goids=function_goids,
        filters=filters,
        component_meta=component_meta,
        runtime_options=runtime_options,
    )


@cache()
def graph_metrics_result(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> GraphMetricsRows:
    """Compute base graph metrics rows.

    Returns
    -------
    GraphMetricsRows
        Container with function and module graph metric rows.
    """
    return build_graph_metrics_rows(
        GraphMetricsInputs(
            snapshot=env.snapshot,
            call_graph=graph_metric_inputs.call_graph,
            import_graph=graph_metric_inputs.import_graph,
            symbol_module_edges=graph_metric_inputs.symbol_module_edges,
            module_names=graph_metric_inputs.module_names,
            component_meta=graph_metric_inputs.component_meta,
            filters=graph_metric_inputs.filters,
            community_detection_limit=(
                graph_metric_inputs.runtime_options.features.community_detection_limit
            ),
            use_gpu=graph_metric_inputs.runtime_options.use_gpu,
        )
    )


def graph_metrics_functions__base(
    graph_metrics_result: GraphMetricsRows,
) -> pa.RecordBatchReader:
    """Build base graph metrics rows for functions.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing function graph metrics rows.
    """
    if not graph_metrics_result.function_rows:
        return empty_reader_for_table(GRAPH_METRICS_FUNCTIONS_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
        graph_metrics_result.function_rows,
    )
    return reader


def graph_metrics_modules__base(
    graph_metrics_result: GraphMetricsRows,
) -> pa.RecordBatchReader:
    """Build base graph metrics rows for modules.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing module graph metrics rows.
    """
    if not graph_metrics_result.module_rows:
        return empty_reader_for_table(GRAPH_METRICS_MODULES_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(
        GRAPH_METRICS_MODULES_TABLE_KEY,
        graph_metrics_result.module_rows,
    )
    return reader


def graph_metrics_functions_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.RecordBatchReader:
    """Build extended graph metrics rows for functions.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing extended function metrics rows.
    """
    rows = build_graph_metrics_functions_ext_rows(
        repo=env.repo,
        commit=env.commit,
        call_graph=graph_metric_inputs.call_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    if not rows:
        return empty_reader_for_table(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY, rows)
    return reader


def graph_metrics_modules_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.RecordBatchReader:
    """Build extended graph metrics rows for modules.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing extended module metrics rows.
    """
    rows = build_graph_metrics_modules_ext_rows(
        repo=env.repo,
        commit=env.commit,
        import_graph=graph_metric_inputs.import_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    if not rows:
        return empty_reader_for_table(GRAPH_METRICS_MODULES_EXT_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(GRAPH_METRICS_MODULES_EXT_TABLE_KEY, rows)
    return reader


def symbol_graph_metrics_functions__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.RecordBatchReader:
    """Build symbol graph metrics rows for functions.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing symbol function metrics rows.
    """
    rows = build_symbol_graph_metrics_function_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_function_graph,
        known_functions=graph_metric_inputs.function_goids or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    if not rows:
        return empty_reader_for_table(SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY, rows)
    return reader


def symbol_graph_metrics_modules__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.RecordBatchReader:
    """Build symbol graph metrics rows for modules.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing symbol module metrics rows.
    """
    rows = build_symbol_graph_metrics_module_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_module_graph,
        known_modules=graph_metric_inputs.module_names or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    if not rows:
        return empty_reader_for_table(SYMBOL_GRAPH_MODULES_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(SYMBOL_GRAPH_MODULES_TABLE_KEY, rows)
    return reader


def graph_stats__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    q__analytics__config_values: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pa.RecordBatchReader:
    """Build base graph stats rows.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader containing graph stats rows.
    """
    config_value_rows = _collect_rows(
        q__analytics__config_values,
        ("repo", "commit", "key", "reference_modules"),
        repo=env.repo,
        commit=env.commit,
    )
    module_rows = _collect_rows(
        q__core__modules,
        ("module", "repo", "commit"),
        repo=env.repo,
        commit=env.commit,
    )
    allowed_modules = _allowed_modules_from_rows(
        module_rows,
        repo=env.repo,
        commit=env.commit,
    )
    config_bipartite = build_config_module_bipartite(
        config_value_rows,
        allowed_modules=allowed_modules,
        repo=env.repo,
        commit=env.commit,
    )
    rows = build_graph_stats_rows(
        GraphStatsInputs(
            repo=env.repo,
            commit=env.commit,
            call_graph=graph_metric_inputs.call_graph,
            import_graph=graph_metric_inputs.import_graph,
            symbol_module_graph=graph_metric_inputs.symbol_module_graph,
            symbol_function_graph=graph_metric_inputs.symbol_function_graph,
            config_module_bipartite=config_bipartite,
            use_gpu=graph_metric_inputs.runtime_options.use_gpu,
        )
    )
    if not rows:
        return empty_reader_for_table(GRAPH_STATS_TABLE_KEY)
    reader, _ = record_batch_reader_for_rows(GRAPH_STATS_TABLE_KEY, rows)
    return reader


_MODULE = sys.modules[__name__]
_GRAPH_METRICS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=GRAPH_METRICS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
            base_node="graph_metrics_functions__base",
            contract=GRAPH_METRICS_FUNCTIONS_CONTRACT,
            save_spec=DatasetSaveSpec(
                table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
                collect_group=GRAPH_METRICS_COLLECT_GROUP,
            ),
            input_type=pa.RecordBatchReader,
            node_name="graph_metrics_functions__table",
        ),
        TableTargetTableSpec(
            table_key=GRAPH_METRICS_MODULES_TABLE_KEY,
            base_node="graph_metrics_modules__base",
            contract=GRAPH_METRICS_MODULES_CONTRACT,
            save_spec=DatasetSaveSpec(
                table_key=GRAPH_METRICS_MODULES_TABLE_KEY,
                collect_group=GRAPH_METRICS_COLLECT_GROUP,
            ),
            input_type=pa.RecordBatchReader,
            node_name="graph_metrics_modules__table",
        ),
    ),
    table_materializations_node="graph_metrics__table_materializations",
    anchor_node_name="t__graph_metrics",
)
attach_table_target_template(_MODULE, spec=_GRAPH_METRICS_TABLE_TARGET_SPEC)
graph_metrics_functions__table = _MODULE.graph_metrics_functions__table
graph_metrics_modules__table = _MODULE.graph_metrics_modules__table
graph_metrics__table_materializations = _MODULE.graph_metrics__table_materializations
t__graph_metrics = _MODULE.t__graph_metrics

_GRAPH_METRICS_EXT_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=GRAPH_METRICS_EXT_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
            base_node="graph_metrics_functions_ext__base",
            contract=GRAPH_METRICS_FUNCTIONS_EXT_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY),
            input_type=pa.RecordBatchReader,
            node_name="graph_metrics_functions_ext__table",
        ),
        TableTargetTableSpec(
            table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
            base_node="graph_metrics_modules_ext__base",
            contract=GRAPH_METRICS_MODULES_EXT_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY),
            input_type=pa.RecordBatchReader,
            node_name="graph_metrics_modules_ext__table",
        ),
    ),
    table_materializations_node="graph_metrics_ext__table_materializations",
    anchor_node_name="t__graph_metrics_ext",
)
attach_table_target_template(_MODULE, spec=_GRAPH_METRICS_EXT_TABLE_TARGET_SPEC)
graph_metrics_functions_ext__table = _MODULE.graph_metrics_functions_ext__table
graph_metrics_modules_ext__table = _MODULE.graph_metrics_modules_ext__table
graph_metrics_ext__table_materializations = _MODULE.graph_metrics_ext__table_materializations
t__graph_metrics_ext = _MODULE.t__graph_metrics_ext

_SYMBOL_GRAPH_METRICS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY,
            base_node="symbol_graph_metrics_functions__base",
            contract=SYMBOL_GRAPH_FUNCTIONS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY),
            input_type=pa.RecordBatchReader,
            node_name="symbol_graph_metrics_functions__table",
        ),
        TableTargetTableSpec(
            table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY,
            base_node="symbol_graph_metrics_modules__base",
            contract=SYMBOL_GRAPH_MODULES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY),
            input_type=pa.RecordBatchReader,
            node_name="symbol_graph_metrics_modules__table",
        ),
    ),
    table_materializations_node="symbol_graph_metrics__table_materializations",
    anchor_node_name="t__symbol_graph_metrics",
)
attach_table_target_template(_MODULE, spec=_SYMBOL_GRAPH_METRICS_TABLE_TARGET_SPEC)
symbol_graph_metrics_functions__table = _MODULE.symbol_graph_metrics_functions__table
symbol_graph_metrics_modules__table = _MODULE.symbol_graph_metrics_modules__table
symbol_graph_metrics__table_materializations = _MODULE.symbol_graph_metrics__table_materializations
t__symbol_graph_metrics = _MODULE.t__symbol_graph_metrics

_GRAPH_STATS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=GRAPH_STATS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=GRAPH_STATS_TABLE_KEY,
            base_node="graph_stats__base",
            contract=GRAPH_STATS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=GRAPH_STATS_TABLE_KEY),
            input_type=pa.RecordBatchReader,
            node_name="graph_stats__table",
        ),
    ),
    table_materializations_node="graph_stats__table_materializations",
    anchor_node_name="t__graph_stats",
)
attach_table_target_template(_MODULE, spec=_GRAPH_STATS_TABLE_TARGET_SPEC)
graph_stats__table = _MODULE.graph_stats__table
graph_stats__table_materializations = _MODULE.graph_stats__table_materializations
t__graph_stats = _MODULE.t__graph_stats


__all__ = [
    "graph_metrics__table_materializations",
    "graph_metrics_ext__table_materializations",
    "graph_metrics_functions__base",
    "graph_metrics_functions__table",
    "graph_metrics_functions_ext__base",
    "graph_metrics_functions_ext__table",
    "graph_metrics_modules__base",
    "graph_metrics_modules__table",
    "graph_metrics_modules_ext__base",
    "graph_metrics_modules_ext__table",
    "graph_stats__base",
    "graph_stats__table",
    "symbol_graph_metrics__table_materializations",
    "symbol_graph_metrics_functions__base",
    "symbol_graph_metrics_functions__table",
    "symbol_graph_metrics_modules__base",
    "symbol_graph_metrics_modules__table",
    "t__graph_metrics",
    "t__graph_metrics_ext",
    "t__graph_stats",
    "t__symbol_graph_metrics",
]
