"""Config graph analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.graphs.config_data_flow import (
    CONFIG_DATA_FLOW_COLS,
    compute_config_data_flow_result,
)
from codeintel.analytics.graphs.config_graph_metrics import (
    CONFIG_GRAPH_METRICS_KEYS_COLS,
    CONFIG_GRAPH_METRICS_MODULES_COLS,
    CONFIG_PROJECTION_KEY_EDGES_COLS,
    CONFIG_PROJECTION_MODULE_EDGES_COLS,
    ConfigGraphMetricsResult,
    compute_config_graph_metrics_result,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
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
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.catalog.service import CatalogService
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

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


def config_data_flow__base(
    env: BuildEnv,
    _q__analytics__config_values: InferableTabularInput,
    _q__graph__call_graph_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build config data flow rows.

    Parameters
    ----------
    env
        Build environment with gateway access.
    _q__analytics__config_values
        Config values input (unused, required for dependency ordering).
    _q__graph__call_graph_edges
        Call graph edges input (unused, required for dependency ordering).

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing config data flow rows.
    """
    runtime = resolve_graph_runtime(
        env.gateway,
        env.snapshot,
        GraphRuntimeOptions(snapshot=env.snapshot),
    )
    catalog = CatalogService.from_db(env.gateway, repo=env.repo, commit=env.commit)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(env.gateway, request)
    result = compute_config_data_flow_result(
        env.gateway,
        env.snapshot,
        call_graph=runtime.ensure_call_graph(),
        ast_by_goid=ast_map,
        missing_goids=missing,
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
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics result rows.

    Returns
    -------
    ConfigGraphMetricsResult
        Computed config graph metrics container.
    """
    return compute_config_graph_metrics_result(env.gateway, repo=env.repo, commit=env.commit)


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
