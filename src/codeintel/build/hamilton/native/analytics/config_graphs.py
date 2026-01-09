"""Config graph analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

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
from codeintel.build.analytics.graphs.config_references import (
    ConfigReferenceInputs,
    compute_config_reference_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.graphs.builders import build_call_graph_from_tables
from codeintel.build.graphs.runtime import GraphRuntimeOptions, graph_runtime_options_from_env
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    TableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import normalize_path
from codeintel.core.spans import normalize_line_span

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CONFIG_REFERENCES_TARGET_NAME = "config_references"
CONFIG_REFERENCES_TABLE_KEY = "analytics.config_references"
CONFIG_REFERENCES_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_REFERENCES_TABLE_KEY,
    target_name=CONFIG_REFERENCES_TARGET_NAME,
    input_name="config_references__base",
    required_cols=(),
    clip_column=None,
)

CONFIG_DATA_FLOW_TARGET_NAME = "config_data_flow"
CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"
CONFIG_DATA_FLOW_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_DATA_FLOW_TABLE_KEY,
    target_name=CONFIG_DATA_FLOW_TARGET_NAME,
    input_name="config_data_flow__base",
    required_cols=(),
    clip_column=None,
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
CONFIG_GRAPH_KEYS_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_GRAPH_KEYS_TABLE_KEY,
    target_name=CONFIG_GRAPH_TARGET_NAME,
    input_name="config_graph_metrics_keys__base",
    required_cols=(),
    clip_column=None,
)
CONFIG_GRAPH_MODULES_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_GRAPH_MODULES_TABLE_KEY,
    target_name=CONFIG_GRAPH_TARGET_NAME,
    input_name="config_graph_metrics_modules__base",
    required_cols=(),
    clip_column=None,
)
CONFIG_GRAPH_KEY_EDGES_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
    target_name=CONFIG_GRAPH_TARGET_NAME,
    input_name="config_projection_key_edges__base",
    required_cols=(),
    clip_column=None,
)
CONFIG_GRAPH_MODULE_EDGES_CONTRACT = contract_ref_for_table(
    table_key=CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
    target_name=CONFIG_GRAPH_TARGET_NAME,
    input_name="config_projection_module_edges__base",
    required_cols=(),
    clip_column=None,
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

    config_references: InferableTabularInput
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


def config_references__base(
    env: BuildEnv,
    q__analytics__config_values: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pa.Table:
    """Build config reference rows.

    Returns
    -------
    pa.Table
        Reader containing config reference rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    config_value_rows = collect_scoped_rows(
        q__analytics__config_values,
        ("repo", "commit", "config_path", "key"),
        scope=scope,
    )
    module_rows = collect_scoped_rows(
        q__core__modules,
        ("path", "module", "language"),
        scope=scope,
        require_scope_columns=False,
    )
    rows = compute_config_reference_rows(
        ConfigReferenceInputs(
            snapshot=env.snapshot,
            config_value_rows=config_value_rows,
            module_rows=module_rows,
        )
    )
    if not rows:
        return empty_table_for_table(CONFIG_REFERENCES_TABLE_KEY)
    return finalize_analytics_rows(CONFIG_REFERENCES_TABLE_KEY, rows)


def config_data_flow_frames(
    config_references__base: InferableTabularInput,
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
        config_references=config_references__base,
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
    config_reference_rows = collect_scoped_rows(
        config_data_flow_frames.config_references,
        ("repo", "commit", "config_path", "key", "extras"),
        scope=scope,
    )
    entrypoint_rows = collect_scoped_rows(
        config_data_flow_frames.entrypoints,
        ("repo", "commit", "handler_goid_h128"),
        scope=scope,
    )
    call_graph_edges = tabular_to_arrow_table(config_data_flow_frames.call_graph_edges)
    call_graph_nodes = tabular_to_arrow_table(config_data_flow_frames.call_graph_nodes)
    call_graph = build_call_graph_from_tables(
        call_graph_edges,
        call_graph_nodes,
        repo=env.repo,
        commit=env.commit,
    )
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
            config_value_rows=config_reference_rows,
            entrypoint_rows=entrypoint_rows,
            call_graph=call_graph,
            ast_by_goid=ast_map,
            missing_goids=missing,
        )
    )
    if result.rows is None:
        return empty_table_for_table(CONFIG_DATA_FLOW_TABLE_KEY)
    return finalize_analytics_rows(CONFIG_DATA_FLOW_TABLE_KEY, result.rows)


@cache(behavior="default")
def config_graph_metrics_result(
    env: BuildEnv,
    config_references__base: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics result rows.

    Returns
    -------
    ConfigGraphMetricsResult
        Computed config graph metrics container.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    config_reference_rows = collect_scoped_rows(
        config_references__base,
        ("repo", "commit", "key", "extras"),
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
        config_value_rows=config_reference_rows,
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
    return finalize_analytics_rows(
        CONFIG_GRAPH_KEYS_TABLE_KEY,
        config_graph_metrics_result.key_rows,
    )


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
    return finalize_analytics_rows(
        CONFIG_GRAPH_MODULES_TABLE_KEY,
        config_graph_metrics_result.module_rows,
    )


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
    return finalize_analytics_rows(
        CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
        config_graph_metrics_result.key_edge_rows,
    )


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
    return finalize_analytics_rows(
        CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
        config_graph_metrics_result.module_edge_rows,
    )


_MODULE = sys.modules[__name__]
_CONFIG_REFERENCES_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=CONFIG_REFERENCES_CONTRACT,
        node_name="config_references__table",
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_CONFIG_REFERENCES_TABLE_TARGET_SPEC)
config_references__table = _MODULE.config_references__table
config_references__table_materializations = _MODULE.config_references__table_materializations
t__config_references = _MODULE.t__config_references

_CONFIG_DATA_FLOW_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=CONFIG_DATA_FLOW_CONTRACT,
        node_name="config_data_flow__table",
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_CONFIG_DATA_FLOW_TABLE_TARGET_SPEC)
config_data_flow__table = _MODULE.config_data_flow__table
config_data_flow__table_materializations = _MODULE.config_data_flow__table_materializations
t__config_data_flow = _MODULE.t__config_data_flow

_CONFIG_GRAPH_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract_ref(
        contract_ref=CONFIG_GRAPH_KEYS_CONTRACT,
        node_name="config_graph_metrics_keys__table",
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=CONFIG_GRAPH_MODULES_CONTRACT,
        node_name="config_graph_metrics_modules__table",
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=CONFIG_GRAPH_KEY_EDGES_CONTRACT,
        node_name="config_projection_key_edges__table",
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=CONFIG_GRAPH_MODULE_EDGES_CONTRACT,
        node_name="config_projection_module_edges__table",
    ),
)
_CONFIG_GRAPH_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=CONFIG_GRAPH_TARGET_NAME,
        tables=(),
        table_materializations_node="config_graph_metrics__table_materializations",
        anchor_node_name="t__config_graph_metrics",
        default_input_type=pa.Table,
    ),
    table_contexts=_CONFIG_GRAPH_TABLE_CONTEXTS,
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
    "config_references__base",
    "config_references__table",
    "config_references__table_materializations",
    "t__config_data_flow",
    "t__config_graph_metrics",
    "t__config_references",
]
