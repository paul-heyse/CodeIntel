"""Config graph analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.build.analytics.graphs.config_data_flow import (
    CONFIG_DATA_FLOW_COLS,
    ConfigDataFlowInputs,
    compute_config_data_flow_result,
)
from codeintel.build.analytics.graphs.config_graph_metrics import (
    CONFIG_GRAPH_METRICS_KEYS_COLS,
    CONFIG_GRAPH_METRICS_MODULES_COLS,
    CONFIG_PROJECTION_KEY_EDGES_COLS,
    CONFIG_PROJECTION_MODULE_EDGES_COLS,
    ConfigGraphMetricsResult,
    compute_config_graph_metrics_result,
)
from codeintel.build.analytics.graphs.graph_metrics import build_call_graph_from_rows
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
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
from codeintel.core.paths import normalize_path

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CONFIG_DATA_FLOW_TARGET_NAME = "config_data_flow"
CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"
CONFIG_DATA_FLOW_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
)
CONFIG_DATA_FLOW_CONTRACT = TableContractSpec(
    table_key=CONFIG_DATA_FLOW_TABLE_KEY,
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="config_data_flow__base",
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
CONFIG_GRAPH_SAVE_CONTEXT = SaverContext(domain="analytics", target=CONFIG_GRAPH_TARGET_NAME)
CONFIG_GRAPH_KEYS_CONTRACT = TableContractSpec(
    table_key=CONFIG_GRAPH_KEYS_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="config_graph_metrics_keys__base",
)
CONFIG_GRAPH_MODULES_CONTRACT = TableContractSpec(
    table_key=CONFIG_GRAPH_MODULES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="config_graph_metrics_modules__base",
)
CONFIG_GRAPH_KEY_EDGES_CONTRACT = TableContractSpec(
    table_key=CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="config_projection_key_edges__base",
)
CONFIG_GRAPH_MODULE_EDGES_CONTRACT = TableContractSpec(
    table_key=CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="config_projection_module_edges__base",
)

_FUNCTION_KINDS: frozenset[str] = frozenset({"function", "method"})


@dataclass(frozen=True)
class _GoidSpan:
    goid: int
    qualname: str
    start_line: int
    end_line: int


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
    rows: Iterable[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, list[_GoidSpan]], set[int]]:
    grouped: dict[str, list[_GoidSpan]] = {}
    missing: set[int] = set()
    for row in rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        kind = row.get("kind")
        if kind is None or str(kind) not in _FUNCTION_KINDS:
            continue
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is None:
            continue
        rel_path = row.get("rel_path")
        start_line = _coerce_int(row.get("start_line"))
        end_line = _coerce_int(row.get("end_line")) or start_line
        if rel_path is None or start_line is None or end_line is None:
            missing.add(goid)
            continue
        qualname = row.get("qualname")
        span = _GoidSpan(
            goid=goid,
            qualname=str(qualname) if qualname is not None else "",
            start_line=start_line,
            end_line=end_line,
        )
        grouped.setdefault(normalize_path(str(rel_path)), []).append(span)
    return grouped, missing


def _function_asts_from_goids(
    rows: Iterable[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
    repo_root: Path,
) -> tuple[dict[int, FunctionAst], set[int]]:
    grouped, missing = _group_goids_by_path(rows, repo=repo, commit=commit)
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


def config_data_flow__base(
    env: BuildEnv,
    _q__analytics__config_values: InferableTabularInput,
    _q__analytics__entrypoints: InferableTabularInput,
    _q__graph__call_graph_edges: InferableTabularInput,
    _q__graph__call_graph_nodes: InferableTabularInput,
    _q__core__goids: InferableTabularInput,
) -> pl.LazyFrame:
    """Build config data flow rows.

    Parameters
    ----------
    env
        Build environment with gateway access.
    _q__analytics__config_values
        Config values input (unused, required for dependency ordering).
    _q__analytics__entrypoints
        Entrypoint rows input (unused, required for dependency ordering).
    _q__graph__call_graph_edges
        Call graph edges input (unused, required for dependency ordering).
    _q__graph__call_graph_nodes
        Call graph nodes input (unused, required for dependency ordering).
    _q__core__goids
        GOID rows for AST lookup (unused, required for dependency ordering).

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing config data flow rows.
    """
    config_value_rows = _collect_rows(
        _q__analytics__config_values,
        ("repo", "commit", "config_path", "key", "reference_paths"),
        repo=env.repo,
        commit=env.commit,
    )
    entrypoint_rows = _collect_rows(
        _q__analytics__entrypoints,
        ("repo", "commit", "handler_goid_h128"),
        repo=env.repo,
        commit=env.commit,
    )
    call_edge_rows = _collect_rows(
        _q__graph__call_graph_edges,
        ("caller_goid_h128", "callee_goid_h128"),
        repo=env.repo,
        commit=env.commit,
    )
    call_node_rows = _collect_rows(
        _q__graph__call_graph_nodes,
        ("goid_h128", "kind"),
        repo=None,
        commit=None,
    )
    call_graph = build_call_graph_from_rows(call_edge_rows, call_node_rows)
    goid_rows = _collect_rows(
        _q__core__goids,
        ("goid_h128", "rel_path", "qualname", "kind", "start_line", "end_line", "repo", "commit"),
        repo=env.repo,
        commit=env.commit,
    )
    ast_map, missing = _function_asts_from_goids(
        goid_rows,
        repo=env.repo,
        commit=env.commit,
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
        return empty_frame_for_table(CONFIG_DATA_FLOW_TABLE_KEY)
    return rows_to_frame(
        CONFIG_DATA_FLOW_TABLE_KEY,
        result.rows,
        columns=CONFIG_DATA_FLOW_COLS,
    )


@save_dataset(
    context=CONFIG_DATA_FLOW_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CONFIG_DATA_FLOW_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=CONFIG_DATA_FLOW_TARGET_NAME,
    table_key=CONFIG_DATA_FLOW_TABLE_KEY,
)
@table_contract(CONFIG_DATA_FLOW_CONTRACT)
def config_data_flow__table(config_data_flow__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist config data flow rows.

    Returns
    -------
    pl.LazyFrame
        Persisted config data flow frame.
    """
    return config_data_flow__base


@codeintel_target(domain="analytics", target=CONFIG_DATA_FLOW_TARGET_NAME)
def t__config_data_flow(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__config_data_flow: MaterializationResult,
) -> TargetRunRecord:
    """Finalize config data flow target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the config data flow target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=CONFIG_DATA_FLOW_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            CONFIG_DATA_FLOW_TABLE_KEY: m__analytics__config_data_flow,
        },
    )


def config_graph_metrics_result(
    env: BuildEnv,
    _q__analytics__config_values: InferableTabularInput,
    _q__core__modules: InferableTabularInput,
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics result rows.

    Returns
    -------
    ConfigGraphMetricsResult
        Computed config graph metrics container.
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
) -> pl.LazyFrame:
    """Build config graph key metrics rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing key metrics rows.
    """
    if config_graph_metrics_result.key_rows is None:
        return empty_frame_for_table(CONFIG_GRAPH_KEYS_TABLE_KEY)
    return rows_to_frame(
        CONFIG_GRAPH_KEYS_TABLE_KEY,
        config_graph_metrics_result.key_rows,
        columns=CONFIG_GRAPH_METRICS_KEYS_COLS,
    )


@save_dataset(
    context=CONFIG_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CONFIG_GRAPH_KEYS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    table_key=CONFIG_GRAPH_KEYS_TABLE_KEY,
)
@table_contract(CONFIG_GRAPH_KEYS_CONTRACT)
def config_graph_metrics_keys__table(config_graph_metrics_keys__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist config graph key metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted key metrics frame.
    """
    return config_graph_metrics_keys__base


def config_graph_metrics_modules__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pl.LazyFrame:
    """Build config graph module metrics rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing module metrics rows.
    """
    if config_graph_metrics_result.module_rows is None:
        return empty_frame_for_table(CONFIG_GRAPH_MODULES_TABLE_KEY)
    return rows_to_frame(
        CONFIG_GRAPH_MODULES_TABLE_KEY,
        config_graph_metrics_result.module_rows,
        columns=CONFIG_GRAPH_METRICS_MODULES_COLS,
    )


@save_dataset(
    context=CONFIG_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CONFIG_GRAPH_MODULES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    table_key=CONFIG_GRAPH_MODULES_TABLE_KEY,
)
@table_contract(CONFIG_GRAPH_MODULES_CONTRACT)
def config_graph_metrics_modules__table(
    config_graph_metrics_modules__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist config graph module metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted module metrics frame.
    """
    return config_graph_metrics_modules__base


def config_projection_key_edges__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pl.LazyFrame:
    """Build config projection key edge rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing config projection key edges.
    """
    if config_graph_metrics_result.key_edge_rows is None:
        return empty_frame_for_table(CONFIG_GRAPH_KEY_EDGES_TABLE_KEY)
    return rows_to_frame(
        CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
        config_graph_metrics_result.key_edge_rows,
        columns=CONFIG_PROJECTION_KEY_EDGES_COLS,
    )


@save_dataset(
    context=CONFIG_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CONFIG_GRAPH_KEY_EDGES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    table_key=CONFIG_GRAPH_KEY_EDGES_TABLE_KEY,
)
@table_contract(CONFIG_GRAPH_KEY_EDGES_CONTRACT)
def config_projection_key_edges__table(
    config_projection_key_edges__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist config projection key edge rows.

    Returns
    -------
    pl.LazyFrame
        Persisted projection key edge frame.
    """
    return config_projection_key_edges__base


def config_projection_module_edges__base(
    config_graph_metrics_result: ConfigGraphMetricsResult,
) -> pl.LazyFrame:
    """Build config projection module edge rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing config projection module edges.
    """
    if config_graph_metrics_result.module_edge_rows is None:
        return empty_frame_for_table(CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY)
    return rows_to_frame(
        CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
        config_graph_metrics_result.module_edge_rows,
        columns=CONFIG_PROJECTION_MODULE_EDGES_COLS,
    )


@save_dataset(
    context=CONFIG_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    table_key=CONFIG_GRAPH_MODULE_EDGES_TABLE_KEY,
)
@table_contract(CONFIG_GRAPH_MODULE_EDGES_CONTRACT)
def config_projection_module_edges__table(
    config_projection_module_edges__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist config projection module edge rows.

    Returns
    -------
    pl.LazyFrame
        Persisted projection module edge frame.
    """
    return config_projection_module_edges__base


config_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=CONFIG_GRAPH_TARGET_NAME,
    table_keys=CONFIG_GRAPH_TABLE_KEYS,
    node_name="config_graph_metrics__table_materializations",
)


@codeintel_target(domain="analytics", target=CONFIG_GRAPH_TARGET_NAME)
def t__config_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    config_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize config graph metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the config graph metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=CONFIG_GRAPH_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=config_graph_metrics__table_materializations,
    )


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
