"""Graph metrics analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import polars as pl
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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.tabular.types import InferableTabularInput
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
) -> pl.DataFrame:
    frame = tabular_to_lazyframe(value)
    available = set(frame.columns)
    if repo is not None and "repo" in available:
        frame = frame.filter(pl.col("repo") == repo)
    if commit is not None and "commit" in available:
        frame = frame.filter(pl.col("commit") == commit)
    return frame.select(list(columns)).collect()


def _matches_optional_scope_expr(column: str, expected: str) -> pl.Expr:
    col = pl.col(column)
    col_str = col.cast(pl.Utf8, strict=False)
    stripped = col_str.str.strip_chars()
    return col.is_null() | (stripped.str.len_chars() == 0) | (col_str == expected)


def _filter_frame_by_snapshot(
    frame: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(_matches_optional_scope_expr("repo", repo))
    if "commit" in filtered.columns:
        filtered = filtered.filter(_matches_optional_scope_expr("commit", commit))
    return filtered


def _allowed_modules_from_frame(
    frame: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> set[str]:
    if frame.is_empty() or "module" not in frame.columns:
        return set()
    filtered = _filter_frame_by_snapshot(frame, repo=repo, commit=commit)
    if filtered.is_empty():
        return set()
    return {str(module) for module in filtered.get_column("module").drop_nulls().to_list()}


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
    return _call_graph_from_frames(call_edge_rows, call_node_rows)


def _load_import_graph(
    env: BuildEnv,
    edges: InferableTabularInput,
    modules: InferableTabularInput,
) -> tuple[nx.DiGraph, dict[str, dict[str, int | bool]] | None]:
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
    return _import_graph_from_frames(import_edge_rows, import_module_rows)


def _call_graph_from_frames(
    edges: pl.DataFrame,
    nodes: pl.DataFrame,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    _add_call_graph_edges(graph, edges)
    _add_call_graph_nodes(graph, nodes)
    return graph


def _add_call_graph_edges(graph: nx.DiGraph, edges: pl.DataFrame) -> None:
    if edges.is_empty():
        return
    if "caller_goid_h128" not in edges.columns or "callee_goid_h128" not in edges.columns:
        return
    callers = edges.get_column("caller_goid_h128").to_list()
    callees = edges.get_column("callee_goid_h128").to_list()
    for caller_raw, callee_raw in zip(callers, callees, strict=True):
        caller = normalize_decimal_id(caller_raw)
        callee = normalize_decimal_id(callee_raw)
        if caller is None or callee is None:
            continue
        if graph.has_edge(caller, callee):
            attrs = graph[caller][callee]
            attrs["weight"] = int(attrs.get("weight", 0)) + 1
        else:
            graph.add_edge(caller, callee, weight=1)


def _add_call_graph_nodes(graph: nx.DiGraph, nodes: pl.DataFrame) -> None:
    if nodes.is_empty() or "goid_h128" not in nodes.columns:
        return
    node_ids = nodes.get_column("goid_h128").to_list()
    kinds = nodes.get_column("kind").to_list() if "kind" in nodes.columns else [None] * len(node_ids)
    for node_raw, kind in zip(node_ids, kinds, strict=True):
        node_id = normalize_decimal_id(node_raw)
        if node_id is None or node_id in graph:
            continue
        attrs: dict[str, object] = {}
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(node_id, **attrs)


def _import_graph_from_frames(
    edges: pl.DataFrame,
    modules: pl.DataFrame,
) -> tuple[nx.DiGraph, ComponentMeta | None]:
    graph = nx.DiGraph()
    fallback_layer_by_module: dict[str, int] = {}
    _add_import_edges(graph, edges, fallback_layer_by_module)
    component_meta = _component_meta_from_import_frame(modules)
    _apply_import_module_frame(graph, modules, fallback_layer_by_module)
    return graph, component_meta


def _add_import_edges(
    graph: nx.DiGraph,
    edges: pl.DataFrame,
    fallback_layer_by_module: dict[str, int],
) -> None:
    if edges.is_empty():
        return
    if "src_module" not in edges.columns or "dst_module" not in edges.columns:
        return
    sources = edges.get_column("src_module").to_list()
    targets = edges.get_column("dst_module").to_list()
    layers = (
        edges.get_column("module_layer").to_list()
        if "module_layer" in edges.columns
        else [None] * len(sources)
    )
    for source_raw, target_raw, layer_raw in zip(sources, targets, layers, strict=True):
        if source_raw is None or target_raw is None:
            continue
        source = str(source_raw)
        target = str(target_raw)
        layer = coerce_optional_int(layer_raw, ctx="module_layer")
        if layer is not None:
            fallback_layer_by_module[source] = layer
        if graph.has_edge(source, target):
            attrs = graph[source][target]
            attrs["weight"] = int(attrs.get("weight", 0)) + 1
        else:
            graph.add_edge(source, target, weight=1)


def _component_meta_from_import_frame(
    frame: pl.DataFrame,
) -> dict[str, dict[str, int | bool]] | None:
    if frame.is_empty() or "module" not in frame.columns:
        return None
    data = frame.select(["module", "scc_id", "component_size", "layer"]).to_dict(
        as_series=False
    )
    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    found = False
    for module, scc_id, component_size, layer in zip(
        data["module"],
        data["scc_id"],
        data["component_size"],
        data["layer"],
        strict=True,
    ):
        if module is None:
            continue
        found = True
        module_name = str(module)
        scc_value = coerce_optional_int(scc_id, ctx="scc_id")
        size_value = coerce_optional_int(component_size, ctx="component_size")
        layer_value = coerce_optional_int(layer, ctx="layer")
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
    frame: pl.DataFrame,
    fallback_layer_by_module: Mapping[str, int],
) -> None:
    if frame.is_empty() or "module" not in frame.columns:
        return
    data = frame.select(["module", "scc_id", "component_size", "layer"]).to_dict(
        as_series=False
    )
    for module, scc_id, component_size, layer in zip(
        data["module"],
        data["scc_id"],
        data["component_size"],
        data["layer"],
        strict=True,
    ):
        if module is None:
            continue
        module_name = str(module)
        attrs: dict[str, object] = {}
        scc_value = coerce_optional_int(scc_id, ctx="scc_id")
        size_value = coerce_optional_int(component_size, ctx="component_size")
        layer_value = coerce_optional_int(layer, ctx="layer")
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


def _symbol_module_edges_from_frame(
    frame: pl.DataFrame,
    module_by_path: Mapping[str, str],
) -> SymbolModuleEdges:
    if frame.is_empty():
        return set(), {}, {}
    mapped = frame.select(
        pl.col("def_path")
        .map_elements(
            lambda value: _map_path_to_module(value, module_by_path),
            return_dtype=pl.Utf8,
        )
        .alias("def_module"),
        pl.col("use_path")
        .map_elements(
            lambda value: _map_path_to_module(value, module_by_path),
            return_dtype=pl.Utf8,
        )
        .alias("use_module"),
    )
    mapped = mapped.drop_nulls(subset=["def_module", "use_module"]).filter(
        pl.col("def_module") != pl.col("use_module")
    )
    if mapped.is_empty():
        return set(), {}, {}
    def_modules = mapped.get_column("def_module").unique().to_list()
    use_modules = mapped.get_column("use_module").unique().to_list()
    modules = {str(module) for module in def_modules + use_modules}
    inbound_rows = mapped.group_by("def_module").agg(pl.col("use_module").unique())
    outbound_rows = mapped.group_by("use_module").agg(pl.col("def_module").unique())
    inbound = {str(row[0]): {str(item) for item in row[1]} for row in inbound_rows.iter_rows()}
    outbound = {str(row[0]): {str(item) for item in row[1]} for row in outbound_rows.iter_rows()}
    return modules, inbound, outbound


def _symbol_module_graph_from_frame(
    frame: pl.DataFrame,
    module_by_path: Mapping[str, str],
) -> nx.Graph:
    graph = nx.Graph()
    if frame.is_empty():
        return graph
    mapped = frame.select(
        pl.col("def_path")
        .map_elements(
            lambda value: _map_path_to_module(value, module_by_path),
            return_dtype=pl.Utf8,
        )
        .alias("def_module"),
        pl.col("use_path")
        .map_elements(
            lambda value: _map_path_to_module(value, module_by_path),
            return_dtype=pl.Utf8,
        )
        .alias("use_module"),
    )
    mapped = mapped.drop_nulls(subset=["def_module", "use_module"]).filter(
        pl.col("def_module") != pl.col("use_module")
    )
    if mapped.is_empty():
        return graph
    edges = mapped.group_by(["use_module", "def_module"]).agg(pl.len().alias("weight"))
    for use_module, def_module, weight in edges.iter_rows():
        graph.add_edge(str(use_module), str(def_module), weight=int(weight))
    return graph


def _symbol_function_graph_from_frame(
    frame: pl.DataFrame,
) -> nx.Graph:
    graph = nx.Graph()
    if frame.is_empty():
        return graph
    if "def_goid_h128" not in frame.columns or "use_goid_h128" not in frame.columns:
        return graph
    normalized = frame.select(
        pl.col("def_goid_h128")
        .map_elements(normalize_decimal_id, return_dtype=pl.Int64)
        .alias("def_goid"),
        pl.col("use_goid_h128")
        .map_elements(normalize_decimal_id, return_dtype=pl.Int64)
        .alias("use_goid"),
    )
    normalized = normalized.drop_nulls(subset=["def_goid", "use_goid"]).filter(
        pl.col("def_goid") != pl.col("use_goid")
    )
    if normalized.is_empty():
        return graph
    edges = normalized.group_by(["use_goid", "def_goid"]).agg(pl.len().alias("weight"))
    for use_goid, def_goid, weight in edges.iter_rows():
        graph.add_edge(int(use_goid), int(def_goid), weight=int(weight))
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
    symbol_module_edges = _symbol_module_edges_from_frame(symbol_rows, module_by_path)
    symbol_module_graph = _symbol_module_graph_from_frame(symbol_rows, module_by_path)
    symbol_function_graph = _symbol_function_graph_from_frame(symbol_rows)
    return symbol_module_edges, symbol_module_graph, symbol_function_graph


def _module_inputs_from_rows(
    rows: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, str], set[str]]:
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    if rows.is_empty():
        return module_by_path, module_names
    filtered = _filter_frame_by_snapshot(rows, repo=repo, commit=commit)
    if filtered.is_empty() or "module" not in filtered.columns:
        return module_by_path, module_names
    module_series = filtered.get_column("module").drop_nulls()
    module_names = {str(module) for module in module_series.to_list()}
    if "path" in filtered.columns:
        paths = filtered.get_column("path").to_list()
        modules = filtered.get_column("module").to_list()
        for path, module in zip(paths, modules, strict=True):
            if path is None or module is None:
                continue
            module_by_path[str(path)] = str(module)
    return module_by_path, module_names


def _function_goids_from_rows(rows: pl.DataFrame) -> set[int]:
    function_goids: set[int] = set()
    if rows.is_empty() or "goid_h128" not in rows.columns:
        return function_goids
    filtered = rows
    if "kind" in rows.columns:
        filtered = rows.filter(pl.col("kind").cast(pl.Utf8, strict=False).is_in(_FUNCTION_KINDS))
    for value in filtered.get_column("goid_h128").to_list():
        goid = normalize_decimal_id(value)
        if goid is not None:
            function_goids.add(goid)
    return function_goids


def _subsystem_ids_from_rows(rows: pl.DataFrame) -> set[str]:
    subsystem_ids: set[str] = set()
    if rows.is_empty() or "subsystem_id" not in rows.columns:
        return subsystem_ids
    for subsystem_id in rows.get_column("subsystem_id").drop_nulls().to_list():
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


def graph_metrics_functions__base(graph_metrics_result: GraphMetricsRows) -> pl.LazyFrame:
    """Build base graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing function graph metrics rows.
    """
    if not graph_metrics_result.function_rows:
        return empty_frame_for_table(GRAPH_METRICS_FUNCTIONS_TABLE_KEY)
    return rows_to_frame(GRAPH_METRICS_FUNCTIONS_TABLE_KEY, graph_metrics_result.function_rows)


def graph_metrics_modules__base(graph_metrics_result: GraphMetricsRows) -> pl.LazyFrame:
    """Build base graph metrics rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing module graph metrics rows.
    """
    if not graph_metrics_result.module_rows:
        return empty_frame_for_table(GRAPH_METRICS_MODULES_TABLE_KEY)
    return rows_to_frame(GRAPH_METRICS_MODULES_TABLE_KEY, graph_metrics_result.module_rows)


def graph_metrics_functions_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pl.LazyFrame:
    """Build extended graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing extended function metrics rows.
    """
    rows = build_graph_metrics_functions_ext_rows(
        repo=env.repo,
        commit=env.commit,
        call_graph=graph_metric_inputs.call_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    return rows_to_frame(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY, rows)


def graph_metrics_modules_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pl.LazyFrame:
    """Build extended graph metrics rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing extended module metrics rows.
    """
    rows = build_graph_metrics_modules_ext_rows(
        repo=env.repo,
        commit=env.commit,
        import_graph=graph_metric_inputs.import_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    return rows_to_frame(GRAPH_METRICS_MODULES_EXT_TABLE_KEY, rows)


def symbol_graph_metrics_functions__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pl.LazyFrame:
    """Build symbol graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing symbol function metrics rows.
    """
    rows = build_symbol_graph_metrics_function_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_function_graph,
        known_functions=graph_metric_inputs.function_goids or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    return rows_to_frame(SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY, rows)


def symbol_graph_metrics_modules__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pl.LazyFrame:
    """Build symbol graph metrics rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing symbol module metrics rows.
    """
    rows = build_symbol_graph_metrics_module_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_module_graph,
        known_modules=graph_metric_inputs.module_names or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    return rows_to_frame(SYMBOL_GRAPH_MODULES_TABLE_KEY, rows)


def graph_stats__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    q__analytics__config_values: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build base graph stats rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing graph stats rows.
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
    allowed_modules = _allowed_modules_from_frame(
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
    return rows_to_frame(GRAPH_STATS_TABLE_KEY, rows)


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
            node_name="graph_metrics_functions_ext__table",
        ),
        TableTargetTableSpec(
            table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
            base_node="graph_metrics_modules_ext__base",
            contract=GRAPH_METRICS_MODULES_EXT_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY),
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
            node_name="symbol_graph_metrics_functions__table",
        ),
        TableTargetTableSpec(
            table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY,
            base_node="symbol_graph_metrics_modules__base",
            contract=SYMBOL_GRAPH_MODULES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY),
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
