"""Graph metrics analytics tables built with inferable tabular nodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import polars as pl

from codeintel.build.analytics.compute.row_builders import (
    build_symbol_module_edges,
    component_metadata_from_import_rows,
)
from codeintel.build.analytics.graphs.config_graph_metrics import (
    build_config_module_bipartite,
)
from codeintel.build.analytics.graphs.graph_metrics import (
    GraphMetricFilters,
    GraphMetricsInputs,
    GraphMetricsRows,
    SymbolModuleEdges,
    build_call_graph_from_rows,
    build_graph_metric_filters_from_sets,
    build_graph_metrics_rows,
    build_import_graph_from_rows,
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
    build_symbol_function_graph,
    build_symbol_graph_metrics_function_rows,
    build_symbol_graph_metrics_module_rows,
    build_symbol_module_graph,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GRAPH_METRICS_TARGET_NAME = "graph_metrics"
GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.graph_metrics_functions"
GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.graph_metrics_modules"
GRAPH_METRICS_TABLE_KEYS = (
    GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    GRAPH_METRICS_MODULES_TABLE_KEY,
)
GRAPH_METRICS_SAVE_CONTEXT = SaverContext(domain="analytics", target=GRAPH_METRICS_TARGET_NAME)
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
GRAPH_METRICS_EXT_TABLE_KEYS = (
    GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
)
GRAPH_METRICS_EXT_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
)
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
SYMBOL_GRAPH_TABLE_KEYS = (
    SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY,
    SYMBOL_GRAPH_MODULES_TABLE_KEY,
)
SYMBOL_GRAPH_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
)
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
GRAPH_STATS_SAVE_CONTEXT = SaverContext(domain="analytics", target=GRAPH_STATS_TARGET_NAME)
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
    component_meta: dict[str, dict[str, int | bool]] | None
    runtime_options: GraphRuntimeOptions


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
    frame = tabular_to_lazyframe(value)
    available = set(frame.columns)
    if repo is not None and "repo" in available:
        frame = frame.filter(pl.col("repo") == repo)
    if commit is not None and "commit" in available:
        frame = frame.filter(pl.col("commit") == commit)
    return frame.select(list(columns)).collect().to_dicts()


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


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
    return build_call_graph_from_rows(call_edge_rows, call_node_rows)


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
    import_graph = build_import_graph_from_rows(import_edge_rows, import_module_rows)
    component_meta = component_metadata_from_import_rows(import_module_rows)
    return import_graph, component_meta


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
    symbol_module_edges = build_symbol_module_edges(symbol_rows, module_by_path)
    symbol_module_graph = build_symbol_module_graph(symbol_rows, module_by_path)
    symbol_function_graph = build_symbol_function_graph(symbol_rows)
    return symbol_module_edges, symbol_module_graph, symbol_function_graph


def _module_inputs_from_rows(
    rows: list[dict[str, object]],
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, str], set[str]]:
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    for row in rows:
        module = row.get("module")
        if module is None:
            continue
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        module_name = str(module)
        module_names.add(module_name)
        path = row.get("path")
        if path is not None:
            module_by_path[str(path)] = module_name
    return module_by_path, module_names


def _function_goids_from_rows(rows: list[dict[str, object]]) -> set[int]:
    function_goids: set[int] = set()
    for row in rows:
        kind = row.get("kind")
        if kind is None or str(kind) not in _FUNCTION_KINDS:
            continue
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is not None:
            function_goids.add(goid)
    return function_goids


def _subsystem_ids_from_rows(rows: list[dict[str, object]]) -> set[str]:
    subsystem_ids: set[str] = set()
    for row in rows:
        subsystem_id = row.get("subsystem_id")
        if subsystem_id is not None:
            subsystem_ids.add(str(subsystem_id))
    return subsystem_ids


def graph_metric_inputs(
    env: BuildEnv,
    _q__core__goids: InferableTabularInput,
    _q__core__modules: InferableTabularInput,
    _q__graph__call_graph_edges: InferableTabularInput,
    _q__graph__call_graph_nodes: InferableTabularInput,
    _q__graph__import_graph_edges: InferableTabularInput,
    _q__graph__import_modules: InferableTabularInput,
    _q__graph__symbol_use_edges: InferableTabularInput,
    _q__analytics__subsystem_modules: InferableTabularInput,
) -> GraphMetricInputs:
    """Assemble shared graph metric inputs from DAG-provided tables.

    Returns
    -------
    GraphMetricInputs
        Structured graph metric inputs for downstream nodes.
    """
    runtime_options = _graph_runtime_options(env)
    module_by_path, module_names = _load_module_inputs(env, _q__core__modules)
    function_goids = _load_function_goids(env, _q__core__goids)
    subsystem_ids = _load_subsystem_ids(env, _q__analytics__subsystem_modules)
    filters = build_graph_metric_filters_from_sets(
        function_goids=function_goids,
        modules=module_names,
        subsystems=subsystem_ids,
    )
    call_graph = _load_call_graph(
        env,
        _q__graph__call_graph_edges,
        _q__graph__call_graph_nodes,
    )
    import_graph, component_meta = _load_import_graph(
        env,
        _q__graph__import_graph_edges,
        _q__graph__import_modules,
    )
    symbol_module_edges, symbol_module_graph, symbol_function_graph = _load_symbol_graphs(
        module_by_path,
        _q__graph__symbol_use_edges,
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


@save_dataset(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_METRICS_TARGET_NAME,
    table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
)
@table_contract(GRAPH_METRICS_FUNCTIONS_CONTRACT)
def graph_metrics_functions__table(
    graph_metrics_functions__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist function graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted function graph metrics frame.
    """
    return graph_metrics_functions__base


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


@save_dataset(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_METRICS_MODULES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_METRICS_TARGET_NAME,
    table_key=GRAPH_METRICS_MODULES_TABLE_KEY,
)
@table_contract(GRAPH_METRICS_MODULES_CONTRACT)
def graph_metrics_modules__table(
    graph_metrics_modules__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist module graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted module graph metrics frame.
    """
    return graph_metrics_modules__base


graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=GRAPH_METRICS_TARGET_NAME,
    table_keys=GRAPH_METRICS_TABLE_KEYS,
    node_name="graph_metrics__table_materializations",
)


@codeintel_target(domain="analytics", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize graph metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the graph metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_METRICS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=graph_metrics__table_materializations,
    )


def graph_metrics_functions_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    _q__analytics__graph_metrics_functions: InferableTabularInput,
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


@save_dataset(
    context=GRAPH_METRICS_EXT_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
    table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
)
@table_contract(GRAPH_METRICS_FUNCTIONS_EXT_CONTRACT)
def graph_metrics_functions_ext__table(
    graph_metrics_functions_ext__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist extended function graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted extended function metrics frame.
    """
    return graph_metrics_functions_ext__base


def graph_metrics_modules_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    _q__analytics__graph_metrics_modules: InferableTabularInput,
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


@save_dataset(
    context=GRAPH_METRICS_EXT_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
    table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
)
@table_contract(GRAPH_METRICS_MODULES_EXT_CONTRACT)
def graph_metrics_modules_ext__table(
    graph_metrics_modules_ext__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist extended module graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted extended module metrics frame.
    """
    return graph_metrics_modules_ext__base


graph_metrics_ext__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=GRAPH_METRICS_EXT_TARGET_NAME,
    table_keys=GRAPH_METRICS_EXT_TABLE_KEYS,
    node_name="graph_metrics_ext__table_materializations",
)


@codeintel_target(domain="analytics", target=GRAPH_METRICS_EXT_TARGET_NAME)
def t__graph_metrics_ext(
    env: BuildEnv,
    catalog: DagCatalog,
    graph_metrics_ext__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize extended graph metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the extended graph metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_METRICS_EXT_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=graph_metrics_ext__table_materializations,
    )


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


@save_dataset(
    context=SYMBOL_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY,
)
@table_contract(SYMBOL_GRAPH_FUNCTIONS_CONTRACT)
def symbol_graph_metrics_functions__table(
    symbol_graph_metrics_functions__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist symbol function graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted symbol function metrics frame.
    """
    return symbol_graph_metrics_functions__base


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


@save_dataset(
    context=SYMBOL_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY,
)
@table_contract(SYMBOL_GRAPH_MODULES_CONTRACT)
def symbol_graph_metrics_modules__table(
    symbol_graph_metrics_modules__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist symbol module graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted symbol module metrics frame.
    """
    return symbol_graph_metrics_modules__base


symbol_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    table_keys=SYMBOL_GRAPH_TABLE_KEYS,
    node_name="symbol_graph_metrics__table_materializations",
)


@codeintel_target(domain="analytics", target=SYMBOL_GRAPH_METRICS_TARGET_NAME)
def t__symbol_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    symbol_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize symbol graph metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the symbol graph metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=symbol_graph_metrics__table_materializations,
    )


def graph_stats__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    _q__analytics__config_values: InferableTabularInput,
    _q__core__modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build base graph stats rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing graph stats rows.
    """
    config_value_rows = _collect_rows(
        _q__analytics__config_values,
        ("repo", "commit", "key", "reference_modules"),
        repo=env.repo,
        commit=env.commit,
    )
    module_rows = _collect_rows(
        _q__core__modules,
        ("module", "repo", "commit"),
        repo=env.repo,
        commit=env.commit,
    )
    allowed_modules = {
        str(row["module"])
        for row in module_rows
        if row.get("module") is not None
        and _matches_optional_scope(row.get("repo"), env.repo)
        and _matches_optional_scope(row.get("commit"), env.commit)
    }
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


@save_dataset(
    context=GRAPH_STATS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_STATS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_STATS_TARGET_NAME,
    table_key=GRAPH_STATS_TABLE_KEY,
)
@table_contract(GRAPH_STATS_CONTRACT)
def graph_stats__table(graph_stats__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist graph stats rows.

    Returns
    -------
    pl.LazyFrame
        Persisted graph stats frame.
    """
    return graph_stats__base


@codeintel_target(domain="analytics", target=GRAPH_STATS_TARGET_NAME)
def t__graph_stats(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__graph_stats: MaterializationResult,
) -> TargetRunRecord:
    """Finalize graph stats target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the graph stats target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_STATS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            GRAPH_STATS_TABLE_KEY: m__analytics__graph_stats,
        },
    )


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
