"""Graph metrics analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

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
from codeintel.build.contracts.registry import contract_for_table
from codeintel.build.graphs.builders import (
    build_call_graph_from_rows,
    build_import_graph_from_rows,
    build_symbol_function_graph,
    build_symbol_module_edges,
    build_symbol_module_graph,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions, graph_runtime_options_from_env
from codeintel.build.graphs.rx.algos import GraphInput
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    MultiTableTargetContext,
    TableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_int

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GRAPH_METRICS_TARGET_NAME = "graph_metrics"
GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.graph_metrics_functions"
GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.graph_metrics_modules"
GRAPH_METRICS_COLLECT_GROUP = "graph_metrics_core"
GRAPH_METRICS_FUNCTIONS_CONTRACT = contract_for_table(
    table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    target_name=GRAPH_METRICS_TARGET_NAME,
    input_name="graph_metrics_functions__base",
    required_cols=(),
    clip_column=None,
)
GRAPH_METRICS_MODULES_CONTRACT = contract_for_table(
    table_key=GRAPH_METRICS_MODULES_TABLE_KEY,
    target_name=GRAPH_METRICS_TARGET_NAME,
    input_name="graph_metrics_modules__base",
    required_cols=(),
    clip_column=None,
)

GRAPH_METRICS_EXT_TARGET_NAME = "graph_metrics_ext"
GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY = "analytics.graph_metrics_functions_ext"
GRAPH_METRICS_MODULES_EXT_TABLE_KEY = "analytics.graph_metrics_modules_ext"
GRAPH_METRICS_FUNCTIONS_EXT_CONTRACT = contract_for_table(
    table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    target_name=GRAPH_METRICS_EXT_TARGET_NAME,
    input_name="graph_metrics_functions_ext__base",
    required_cols=(),
    clip_column=None,
)
GRAPH_METRICS_MODULES_EXT_CONTRACT = contract_for_table(
    table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
    target_name=GRAPH_METRICS_EXT_TARGET_NAME,
    input_name="graph_metrics_modules_ext__base",
    required_cols=(),
    clip_column=None,
)

SYMBOL_GRAPH_METRICS_TARGET_NAME = "symbol_graph_metrics"
SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY = "analytics.symbol_graph_metrics_functions"
SYMBOL_GRAPH_MODULES_TABLE_KEY = "analytics.symbol_graph_metrics_modules"
SYMBOL_GRAPH_FUNCTIONS_CONTRACT = contract_for_table(
    table_key=SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY,
    target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    input_name="symbol_graph_metrics_functions__base",
    required_cols=(),
    clip_column=None,
)
SYMBOL_GRAPH_MODULES_CONTRACT = contract_for_table(
    table_key=SYMBOL_GRAPH_MODULES_TABLE_KEY,
    target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    input_name="symbol_graph_metrics_modules__base",
    required_cols=(),
    clip_column=None,
)

GRAPH_STATS_TARGET_NAME = "graph_stats"
GRAPH_STATS_TABLE_KEY = "analytics.graph_stats"
GRAPH_STATS_CONTRACT = contract_for_table(
    table_key=GRAPH_STATS_TABLE_KEY,
    target_name=GRAPH_STATS_TARGET_NAME,
    input_name="graph_stats__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True)
class GraphMetricInputs:
    """Shared graph metric inputs derived from DAG sources."""

    call_graph: GraphInput
    import_graph: GraphInput
    symbol_module_edges: SymbolModuleEdges
    symbol_module_graph: GraphInput
    symbol_function_graph: GraphInput
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


@cache(behavior="ignore")
def _graph_runtime_options(env: BuildEnv) -> GraphRuntimeOptions:
    return graph_runtime_options_from_env(env)


def _allowed_modules_from_rows(
    rows: list[dict[str, object]],
    *,
    scope: SnapshotScope,
) -> set[str]:
    if not rows:
        return set()
    filtered = scope.filter_rows(rows, require_keys=True)
    if not filtered:
        return set()
    return {str(row.get("module")) for row in filtered if row.get("module") is not None}


def _load_module_inputs(
    env: BuildEnv, table: InferableTabularInput
) -> tuple[dict[str, str], set[str]]:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    modules_rows = collect_scoped_rows(
        table,
        ("module", "path", "repo", "commit"),
        scope=scope,
    )
    return _module_inputs_from_rows(
        modules_rows,
        scope=scope,
    )


def _load_function_goids(env: BuildEnv, table: InferableTabularInput) -> set[int]:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goid_rows = collect_scoped_rows(
        table,
        ("goid_h128", "kind"),
        scope=scope,
    )
    return _function_goids_from_rows(goid_rows)


def _load_subsystem_ids(env: BuildEnv, table: InferableTabularInput) -> set[str]:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    subsystem_rows = collect_scoped_rows(
        table,
        ("subsystem_id",),
        scope=scope,
    )
    return _subsystem_ids_from_rows(subsystem_rows)


def _load_call_graph(
    env: BuildEnv,
    edges: InferableTabularInput,
    nodes: InferableTabularInput,
) -> GraphInput:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    call_edge_rows = collect_scoped_rows(
        edges,
        ("caller_goid_h128", "callee_goid_h128"),
        scope=scope,
    )
    call_node_rows = collect_scoped_rows(
        nodes,
        ("goid_h128", "kind"),
        scope=scope,
        require_scope_columns=False,
    )
    return _call_graph_from_rows(call_edge_rows, call_node_rows)


def _load_import_graph(
    env: BuildEnv,
    edges: InferableTabularInput,
    modules: InferableTabularInput,
) -> tuple[GraphInput, ComponentMeta | None]:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    import_edge_rows = collect_scoped_rows(
        edges,
        ("src_module", "dst_module", "module_layer"),
        scope=scope,
    )
    import_module_rows = collect_scoped_rows(
        modules,
        ("module", "scc_id", "component_size", "layer"),
        scope=scope,
    )
    return _import_graph_from_rows(import_edge_rows, import_module_rows)


def _call_graph_from_rows(
    edges: list[dict[str, object]],
    nodes: list[dict[str, object]],
) -> GraphInput:
    return build_call_graph_from_rows(edges, nodes)


def _import_graph_from_rows(
    edges: list[dict[str, object]],
    modules: list[dict[str, object]],
) -> tuple[GraphInput, ComponentMeta | None]:
    graph = build_import_graph_from_rows(edges, modules)
    component_meta = _component_meta_from_import_rows(modules)
    return graph, component_meta


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


def _load_symbol_graphs(
    module_by_path: Mapping[str, str],
    table: InferableTabularInput,
    *,
    scope: SnapshotScope,
) -> tuple[SymbolModuleEdges, GraphInput, GraphInput]:
    symbol_rows = collect_scoped_rows(
        table,
        ("def_path", "use_path", "def_goid_h128", "use_goid_h128"),
        scope=scope,
        require_scope_columns=False,
    )
    symbol_module_edges = build_symbol_module_edges(symbol_rows, module_by_path)
    symbol_module_graph = build_symbol_module_graph(
        symbol_rows,
        module_by_path,
    )
    symbol_function_graph = build_symbol_function_graph(
        symbol_rows,
    )
    return symbol_module_edges, symbol_module_graph, symbol_function_graph


def _module_inputs_from_rows(
    rows: list[dict[str, object]],
    *,
    scope: SnapshotScope,
) -> tuple[dict[str, str], set[str]]:
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    if not rows:
        return module_by_path, module_names
    filtered = scope.filter_rows(rows, require_keys=True)
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
    scope = SnapshotScope.from_snapshot(env.snapshot)
    symbol_module_edges, symbol_module_graph, symbol_function_graph = _load_symbol_graphs(
        module_by_path,
        graph_metric_support_frames.symbol_use_edges,
        scope=scope,
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


@cache(behavior="default")
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
) -> pa.Table:
    """Build base graph metrics rows for functions.

    Returns
    -------
    pyarrow.Table
        Table containing function graph metrics rows.
    """
    if not graph_metrics_result.function_rows:
        return empty_table_for_table(GRAPH_METRICS_FUNCTIONS_TABLE_KEY)
    table, _ = table_for_rows(
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
        graph_metrics_result.function_rows,
    )
    return table


def graph_metrics_modules__base(
    graph_metrics_result: GraphMetricsRows,
) -> pa.Table:
    """Build base graph metrics rows for modules.

    Returns
    -------
    pyarrow.Table
        Table containing module graph metrics rows.
    """
    if not graph_metrics_result.module_rows:
        return empty_table_for_table(GRAPH_METRICS_MODULES_TABLE_KEY)
    table, _ = table_for_rows(
        GRAPH_METRICS_MODULES_TABLE_KEY,
        graph_metrics_result.module_rows,
    )
    return table


def graph_metrics_functions_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.Table:
    """Build extended graph metrics rows for functions.

    Returns
    -------
    pyarrow.Table
        Table containing extended function metrics rows.
    """
    rows = build_graph_metrics_functions_ext_rows(
        repo=env.repo,
        commit=env.commit,
        call_graph=graph_metric_inputs.call_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    if not rows:
        return empty_table_for_table(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY)
    table, _ = table_for_rows(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY, rows)
    return table


def graph_metrics_modules_ext__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.Table:
    """Build extended graph metrics rows for modules.

    Returns
    -------
    pyarrow.Table
        Table containing extended module metrics rows.
    """
    rows = build_graph_metrics_modules_ext_rows(
        repo=env.repo,
        commit=env.commit,
        import_graph=graph_metric_inputs.import_graph,
        runtime=graph_metric_inputs.runtime_options,
        filters=graph_metric_inputs.filters,
    )
    if not rows:
        return empty_table_for_table(GRAPH_METRICS_MODULES_EXT_TABLE_KEY)
    table, _ = table_for_rows(GRAPH_METRICS_MODULES_EXT_TABLE_KEY, rows)
    return table


def symbol_graph_metrics_functions__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.Table:
    """Build symbol graph metrics rows for functions.

    Returns
    -------
    pyarrow.Table
        Table containing symbol function metrics rows.
    """
    rows = build_symbol_graph_metrics_function_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_function_graph,
        known_functions=graph_metric_inputs.function_goids or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    if not rows:
        return empty_table_for_table(SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY)
    table, _ = table_for_rows(SYMBOL_GRAPH_FUNCTIONS_TABLE_KEY, rows)
    return table


def symbol_graph_metrics_modules__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
) -> pa.Table:
    """Build symbol graph metrics rows for modules.

    Returns
    -------
    pyarrow.Table
        Table containing symbol module metrics rows.
    """
    rows = build_symbol_graph_metrics_module_rows(
        repo=env.repo,
        commit=env.commit,
        graph=graph_metric_inputs.symbol_module_graph,
        known_modules=graph_metric_inputs.module_names or None,
        runtime=graph_metric_inputs.runtime_options,
    )
    if not rows:
        return empty_table_for_table(SYMBOL_GRAPH_MODULES_TABLE_KEY)
    table, _ = table_for_rows(SYMBOL_GRAPH_MODULES_TABLE_KEY, rows)
    return table


def graph_stats__base(
    env: BuildEnv,
    graph_metric_inputs: GraphMetricInputs,
    q__analytics__config_values: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pa.Table:
    """Build base graph stats rows.

    Returns
    -------
    pyarrow.Table
        Table containing graph stats rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    config_value_rows = collect_scoped_rows(
        q__analytics__config_values,
        ("repo", "commit", "key", "reference_modules"),
        scope=scope,
    )
    module_rows = collect_scoped_rows(
        q__core__modules,
        ("module", "repo", "commit"),
        scope=scope,
    )
    allowed_modules = _allowed_modules_from_rows(
        module_rows,
        scope=scope,
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
        return empty_table_for_table(GRAPH_STATS_TABLE_KEY)
    reader, _ = table_for_rows(GRAPH_STATS_TABLE_KEY, rows)
    return reader


def _graph_metrics_save_spec(table_key: str) -> DatasetSaveSpec:
    return DatasetSaveSpec(
        table_key=table_key,
        collect_group=GRAPH_METRICS_COLLECT_GROUP,
    )


_MODULE = sys.modules[__name__]
_GRAPH_METRICS_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract(
        contract=GRAPH_METRICS_FUNCTIONS_CONTRACT,
        node_name="graph_metrics_functions__table",
    ),
    TableTargetTableContext.from_contract(
        contract=GRAPH_METRICS_MODULES_CONTRACT,
        node_name="graph_metrics_modules__table",
    ),
)
_GRAPH_METRICS_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=GRAPH_METRICS_TARGET_NAME,
        tables=(),
        table_materializations_node="graph_metrics__table_materializations",
        anchor_node_name="t__graph_metrics",
        save_spec_factory=_graph_metrics_save_spec,
        default_input_type=pa.Table,
    ),
    table_contexts=_GRAPH_METRICS_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_GRAPH_METRICS_TABLE_TARGET_SPEC)
graph_metrics_functions__table = _MODULE.graph_metrics_functions__table
graph_metrics_modules__table = _MODULE.graph_metrics_modules__table
graph_metrics__table_materializations = _MODULE.graph_metrics__table_materializations
t__graph_metrics = _MODULE.t__graph_metrics

_GRAPH_METRICS_EXT_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract(
        contract=GRAPH_METRICS_FUNCTIONS_EXT_CONTRACT,
        node_name="graph_metrics_functions_ext__table",
    ),
    TableTargetTableContext.from_contract(
        contract=GRAPH_METRICS_MODULES_EXT_CONTRACT,
        node_name="graph_metrics_modules_ext__table",
    ),
)
_GRAPH_METRICS_EXT_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=GRAPH_METRICS_EXT_TARGET_NAME,
        tables=(),
        table_materializations_node="graph_metrics_ext__table_materializations",
        anchor_node_name="t__graph_metrics_ext",
        default_input_type=pa.Table,
    ),
    table_contexts=_GRAPH_METRICS_EXT_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_GRAPH_METRICS_EXT_TABLE_TARGET_SPEC)
graph_metrics_functions_ext__table = _MODULE.graph_metrics_functions_ext__table
graph_metrics_modules_ext__table = _MODULE.graph_metrics_modules_ext__table
graph_metrics_ext__table_materializations = _MODULE.graph_metrics_ext__table_materializations
t__graph_metrics_ext = _MODULE.t__graph_metrics_ext

_SYMBOL_GRAPH_METRICS_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract(
        contract=SYMBOL_GRAPH_FUNCTIONS_CONTRACT,
        node_name="symbol_graph_metrics_functions__table",
    ),
    TableTargetTableContext.from_contract(
        contract=SYMBOL_GRAPH_MODULES_CONTRACT,
        node_name="symbol_graph_metrics_modules__table",
    ),
)
_SYMBOL_GRAPH_METRICS_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
        tables=(),
        table_materializations_node="symbol_graph_metrics__table_materializations",
        anchor_node_name="t__symbol_graph_metrics",
        default_input_type=pa.Table,
    ),
    table_contexts=_SYMBOL_GRAPH_METRICS_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_SYMBOL_GRAPH_METRICS_TABLE_TARGET_SPEC)
symbol_graph_metrics_functions__table = _MODULE.symbol_graph_metrics_functions__table
symbol_graph_metrics_modules__table = _MODULE.symbol_graph_metrics_modules__table
symbol_graph_metrics__table_materializations = _MODULE.symbol_graph_metrics__table_materializations
t__symbol_graph_metrics = _MODULE.t__symbol_graph_metrics

_GRAPH_STATS_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=GRAPH_STATS_CONTRACT,
        input_type=pa.Table,
    )
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
