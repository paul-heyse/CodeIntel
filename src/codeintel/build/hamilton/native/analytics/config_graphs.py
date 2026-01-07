"""Config graph analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pyarrow as pa
from hamilton.function_modifiers import cache

from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.build.analytics.graphs.config_data_flow import (
    ConfigDataFlowInputs,
    compute_config_data_flow_result,
)
from codeintel.build.analytics.graphs.config_graph_metrics import (
    ConfigGraphMetricsResult,
    compute_config_graph_metrics_result,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.build.contracts.registry import require_contract
from codeintel.build.contracts.types import ContractOverrides
from codeintel.build.graphs.runtime import GraphRuntimeOptions, graph_runtime_options_from_env
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    TableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import normalize_path
from codeintel.core.spans import normalize_line_span

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CONFIG_DATA_FLOW_TARGET_NAME = "config_data_flow"
CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"
CONFIG_DATA_FLOW_CONTRACT = require_contract(
    table_key=CONFIG_DATA_FLOW_TABLE_KEY,
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="config_data_flow__base",
        required_cols=(),
        clip_column=None,
    ),
)

CONFIG_GRAPH_TARGET_NAME = "config_graph_metrics"
CONFIG_GRAPH_KEYS_TABLE_KEY = "analytics.config_graph_metrics_keys"
CONFIG_GRAPH_MODULES_TABLE_KEY = "analytics.config_graph_metrics_modules"
CONFIG_GRAPH_KEY_EDGES_TABLE_KEY = "analytics.config_projection_key_edges"
CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY = "analytics.config_projection_module_edges"
CONFIG_GRAPH_TABLE_KEYS = (
    CONFIG_GRAPH_KEYS_TABLE_KEY,
    CONFIG_GRAPH_MODULES_TABLE_KEY,
    CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
    CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
)
CONFIG_GRAPH_KEYS_CONTRACT = require_contract(
    table_key=CONFIG_GRAPH_KEYS_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="config_graph_metrics_keys__base",
        required_cols=(),
        clip_column=None,
    ),
)
CONFIG_GRAPH_MODULES_CONTRACT = require_contract(
    table_key=CONFIG_GRAPH_MODULES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="config_graph_metrics_modules__base",
        required_cols=(),
        clip_column=None,
    ),
)
CONFIG_GRAPH_KEY_EDGES_CONTRACT = require_contract(
    table_key=CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="config_projection_key_edges__base",
        required_cols=(),
        clip_column=None,
    ),
)
CONFIG_GRAPH_MODULE_EDGES_CONTRACT = require_contract(
    table_key=CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="config_projection_module_edges__base",
        required_cols=(),
        clip_column=None,
    ),
)

_FUNCTION_KINDS: frozenset[str] = frozenset({"function", "method"})


@dataclass(frozen=True)
class _GoidSpan:
    goid: int
    qualname: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ConfigDataFlowFrames:
    """Tabular inputs needed for config data flow computation."""

    config_values: InferableTabularInput
    entrypoints: InferableTabularInput
    call_graph_edges: InferableTabularInput
    call_graph_nodes: InferableTabularInput
    goids: InferableTabularInput


@cache(behavior="ignore")
def _graph_runtime_options(env: BuildEnv) -> GraphRuntimeOptions:
    return graph_runtime_options_from_env(env)


def _allowed_modules_from_frame(
    rows: list[dict[str, object]],
    *,
    scope: SnapshotScope,
) -> set[str]:
    if not rows:
        return set()
    filtered = scope.filter_rows(rows, require_keys=True)
    if not filtered:
        return set()
    return {str(row["module"]) for row in filtered if row.get("module") is not None}


def _coerce_int(value: object) -> int | None:
    parsed: int | None = None
    if value is None:
        parsed = None
    elif isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value) if value.is_integer() else None
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            parsed = None
    return parsed


def _group_goids_by_path(
    rows: list[dict[str, object]],
    *,
    scope: SnapshotScope,
) -> tuple[dict[str, list[_GoidSpan]], set[int]]:
    grouped: dict[str, list[_GoidSpan]] = {}
    missing: set[int] = set()
    if not rows:
        return grouped, missing
    filtered = scope.filter_rows(rows, require_keys=True)
    for row in filtered:
        kind = row.get("kind")
        if kind is None or str(kind) not in _FUNCTION_KINDS:
            continue
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is None:
            continue
        rel_path = row.get("rel_path")
        if rel_path is None:
            missing.add(goid)
            continue
        start_line = _coerce_int(row.get("start_line"))
        end_line = _coerce_int(row.get("end_line"))
        if start_line is None:
            missing.add(goid)
            continue
        start_line, end_line = normalize_line_span(start_line, end_line)
        span = _GoidSpan(
            goid=goid,
            qualname=str(row.get("qualname") or ""),
            start_line=start_line,
            end_line=end_line,
        )
        grouped.setdefault(normalize_path(str(rel_path)), []).append(span)
    return grouped, missing


def _call_graph_from_frames(
    edges: list[dict[str, object]],
    nodes: list[dict[str, object]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    _add_call_graph_edges(graph, edges)
    _add_call_graph_nodes(graph, nodes)
    return graph


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
            current_weight = _coerce_int(attrs.get("weight"))
            attrs["weight"] = (current_weight or 0) + 1
        else:
            graph.add_edge(caller, callee, weight=1)


def _add_call_graph_nodes(graph: nx.DiGraph, nodes: list[dict[str, object]]) -> None:
    if not nodes:
        return
    for row in nodes:
        node = normalize_decimal_id(row.get("goid_h128"))
        if node is None or node in graph:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(node, **attrs)


def _function_asts_from_goids(
    rows: list[dict[str, object]],
    *,
    scope: SnapshotScope,
    repo_root: Path,
) -> tuple[dict[int, FunctionAst], set[int]]:
    grouped, missing = _group_goids_by_path(rows, scope=scope)
    ast_by_goid: dict[int, FunctionAst] = {}
    for rel_path, spans in grouped.items():
        abs_path = (repo_root / rel_path).resolve()
        try:
            parsed = parse_python_file(abs_path)
        except (OSError, ValueError):
            missing.update(span.goid for span in spans)
            continue
        for span in spans:
            node = parsed.span_index.lookup(span.start_line, span.end_line)
            if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                missing.add(span.goid)
                continue
            ast_by_goid[span.goid] = FunctionAst(
                goid=span.goid,
                rel_path=rel_path,
                qualname=span.qualname,
                start_line=span.start_line,
                end_line=span.end_line,
                node=node,
                lines=list(parsed.lines),
            )
    return ast_by_goid, missing


def config_data_flow_frames(
    q__analytics__config_values: InferableTabularInput,
    q__analytics__entrypoints: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__call_graph_nodes: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> ConfigDataFlowFrames:
    """Bundle DAG-provided tables for config data flow computation.

    Returns
    -------
    ConfigDataFlowFrames
        Bundled frame inputs for config data flow computation.
    """
    return ConfigDataFlowFrames(
        config_values=q__analytics__config_values,
        entrypoints=q__analytics__entrypoints,
        call_graph_edges=q__graph__call_graph_edges,
        call_graph_nodes=q__graph__call_graph_nodes,
        goids=q__core__goids,
    )


def config_data_flow__base(
    env: BuildEnv,
    config_data_flow_frames: ConfigDataFlowFrames,
) -> pa.Table:
    """Build config data flow rows.

    Parameters
    ----------
    env
        Build environment with gateway access.
    config_data_flow_frames
        Bundled tabular inputs required for dependency ordering.

    Returns
    -------
    pa.Table
        Reader containing config data flow rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    config_value_rows = collect_scoped_rows(
        config_data_flow_frames.config_values,
        ("repo", "commit", "config_path", "key", "reference_paths"),
        scope=scope,
    )
    entrypoint_rows = collect_scoped_rows(
        config_data_flow_frames.entrypoints,
        ("repo", "commit", "handler_goid_h128"),
        scope=scope,
    )
    call_edge_rows = collect_scoped_rows(
        config_data_flow_frames.call_graph_edges,
        ("caller_goid_h128", "callee_goid_h128"),
        scope=scope,
    )
    call_node_rows = collect_scoped_rows(
        config_data_flow_frames.call_graph_nodes,
        ("goid_h128", "kind"),
        scope=scope,
        require_scope_columns=False,
    )
    call_graph = _call_graph_from_frames(call_edge_rows, call_node_rows)
    goid_rows = collect_scoped_rows(
        config_data_flow_frames.goids,
        ("goid_h128", "rel_path", "qualname", "kind", "start_line", "end_line", "repo", "commit"),
        scope=scope,
    )
    ast_map, missing = _function_asts_from_goids(
        goid_rows,
        scope=scope,
        repo_root=env.snapshot.repo_root,
    )
    result = compute_config_data_flow_result(
        ConfigDataFlowInputs(
            snapshot=env.snapshot,
            config_value_rows=config_value_rows,
            entrypoint_rows=entrypoint_rows,
            call_graph=call_graph,
            ast_by_goid=ast_map,
            missing_goids=missing,
        )
    )
    if result.rows is None:
        return empty_table_for_table(CONFIG_DATA_FLOW_TABLE_KEY)
    reader, _ = table_for_rows(CONFIG_DATA_FLOW_TABLE_KEY, result.rows)
    return reader


@cache(behavior="default")
def config_graph_metrics_result(
    env: BuildEnv,
    q__analytics__config_values: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics result rows.

    Returns
    -------
    ConfigGraphMetricsResult
        Computed config graph metrics container.
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
    allowed_modules = _allowed_modules_from_frame(
        module_rows,
        scope=scope,
    )
    runtime_options = _graph_runtime_options(env)
    return compute_config_graph_metrics_result(
        repo=env.repo,
        commit=env.commit,
        config_value_rows=config_value_rows,
        allowed_modules=allowed_modules,
        runtime=runtime_options,
    )


def config_graph_metrics_keys__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pa.Table:
    """Build config graph key metrics rows.

    Returns
    -------
    pa.Table
        Reader containing key metrics rows.
    """
    if config_graph_metrics_result.key_rows is None:
        return empty_table_for_table(CONFIG_GRAPH_KEYS_TABLE_KEY)
    reader, _ = table_for_rows(
        CONFIG_GRAPH_KEYS_TABLE_KEY,
        config_graph_metrics_result.key_rows,
    )
    return reader


def config_graph_metrics_modules__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pa.Table:
    """Build config graph module metrics rows.

    Returns
    -------
    pa.Table
        Reader containing module metrics rows.
    """
    if config_graph_metrics_result.module_rows is None:
        return empty_table_for_table(CONFIG_GRAPH_MODULES_TABLE_KEY)
    reader, _ = table_for_rows(
        CONFIG_GRAPH_MODULES_TABLE_KEY,
        config_graph_metrics_result.module_rows,
    )
    return reader


def config_projection_key_edges__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pa.Table:
    """Build config projection key edge rows.

    Returns
    -------
    pa.Table
        Reader containing config projection key edges.
    """
    if config_graph_metrics_result.key_edge_rows is None:
        return empty_table_for_table(CONFIG_GRAPH_KEY_EDGES_TABLE_KEY)
    reader, _ = table_for_rows(
        CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
        config_graph_metrics_result.key_edge_rows,
    )
    return reader


def config_projection_module_edges__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pa.Table:
    """Build config projection module edge rows.

    Returns
    -------
    pa.Table
        Reader containing config projection module edges.
    """
    if config_graph_metrics_result.module_edge_rows is None:
        return empty_table_for_table(CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY)
    reader, _ = table_for_rows(
        CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
        config_graph_metrics_result.module_edge_rows,
    )
    return reader


_MODULE = sys.modules[__name__]
_CONFIG_DATA_FLOW_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=CONFIG_DATA_FLOW_CONTRACT,
        node_name="config_data_flow__table",
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_CONFIG_DATA_FLOW_TABLE_TARGET_SPEC)
config_data_flow__table = _MODULE.config_data_flow__table
config_data_flow__table_materializations = _MODULE.config_data_flow__table_materializations
t__config_data_flow = _MODULE.t__config_data_flow

_CONFIG_GRAPH_TABLE_TARGET_SPEC = build_multi_table_target_spec(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=CONFIG_GRAPH_TARGET_NAME,
        tables=(
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext.from_contract(
                    contract=CONFIG_GRAPH_KEYS_CONTRACT,
                    node_name="config_graph_metrics_keys__table",
                    input_type=pa.Table,
                ),
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext.from_contract(
                    contract=CONFIG_GRAPH_MODULES_CONTRACT,
                    node_name="config_graph_metrics_modules__table",
                    input_type=pa.Table,
                ),
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext.from_contract(
                    contract=CONFIG_GRAPH_KEY_EDGES_CONTRACT,
                    node_name="config_projection_key_edges__table",
                    input_type=pa.Table,
                ),
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext.from_contract(
                    contract=CONFIG_GRAPH_MODULE_EDGES_CONTRACT,
                    node_name="config_projection_module_edges__table",
                    input_type=pa.Table,
                ),
            ),
        ),
        table_materializations_node="config_graph_metrics__table_materializations",
        anchor_node_name="t__config_graph_metrics",
    )
)
attach_table_target_template(_MODULE, spec=_CONFIG_GRAPH_TABLE_TARGET_SPEC)
config_graph_metrics_keys__table = _MODULE.config_graph_metrics_keys__table
config_graph_metrics_modules__table = _MODULE.config_graph_metrics_modules__table
config_projection_key_edges__table = _MODULE.config_projection_key_edges__table
config_projection_module_edges__table = _MODULE.config_projection_module_edges__table
config_graph_metrics__table_materializations = _MODULE.config_graph_metrics__table_materializations
t__config_graph_metrics = _MODULE.t__config_graph_metrics


__all__ = [
    "config_data_flow__base",
    "config_data_flow__table",
    "config_graph_metrics__table_materializations",
    "config_graph_metrics_keys__base",
    "config_graph_metrics_keys__table",
    "config_graph_metrics_modules__base",
    "config_graph_metrics_modules__table",
    "config_projection_key_edges__base",
    "config_projection_key_edges__table",
    "config_projection_module_edges__base",
    "config_projection_module_edges__table",
    "t__config_data_flow",
    "t__config_graph_metrics",
]
