"""Graph metrics analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.graphs.graph_metrics import GraphMetricsRows, build_graph_metrics_rows
from codeintel.analytics.graphs.graph_metrics_ext import build_graph_metrics_functions_ext_rows
from codeintel.analytics.graphs.graph_stats import build_graph_stats_rows
from codeintel.analytics.graphs.module_graph_metrics_ext import (
    build_graph_metrics_modules_ext_rows,
)
from codeintel.analytics.graphs.symbol_graph_metrics import (
    build_symbol_graph_metrics_function_rows,
    build_symbol_graph_metrics_module_rows,
)
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


def graph_metrics_result(
    env: BuildEnv,
    _q__graph__call_graph_edges: InferableTabularInput,
    _q__graph__import_graph_edges: InferableTabularInput,
    _q__graph__symbol_use_edges: InferableTabularInput,
    _q__analytics__subsystem_modules: InferableTabularInput,
) -> GraphMetricsRows:
    """Compute base graph metrics rows.

    Returns
    -------
    GraphMetricsRows
        Container with function and module graph metric rows.
    """
    return build_graph_metrics_rows(env.gateway, env.snapshot)


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
    _q__analytics__graph_metrics_functions: InferableTabularInput,
) -> pl.LazyFrame:
    """Build extended graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing extended function metrics rows.
    """
    rows = build_graph_metrics_functions_ext_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
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
    _q__analytics__graph_metrics_modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build extended graph metrics rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing extended module metrics rows.
    """
    rows = build_graph_metrics_modules_ext_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
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
    _q__graph__symbol_use_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build symbol graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing symbol function metrics rows.
    """
    rows = build_symbol_graph_metrics_function_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
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
    _q__graph__symbol_use_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build symbol graph metrics rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing symbol module metrics rows.
    """
    rows = build_symbol_graph_metrics_module_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
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
    _q__graph__call_graph_edges: InferableTabularInput,
    _q__graph__import_graph_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build base graph stats rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing graph stats rows.
    """
    rows = build_graph_stats_rows(env.gateway, repo=env.repo, commit=env.commit)
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
