"""Data model analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.compute.data_models.usage import build_data_model_usage_rows
from codeintel.analytics.data_models.compute import (
    DataModelsResult,
    compute_data_models_from_inputs,
    load_data_models_inputs,
)
from codeintel.analytics.data_models.core import (
    DATA_MODEL_FIELDS_COLS,
    DATA_MODEL_RELATIONSHIPS_COLS,
    DATA_MODELS_COLS,
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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.catalog.service import CatalogService

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

DATA_MODELS_TARGET_NAME = "data_models"
DATA_MODELS_TABLE_KEY = "analytics.data_models"
DATA_MODEL_FIELDS_TABLE_KEY = "analytics.data_model_fields"
DATA_MODEL_RELATIONSHIPS_TABLE_KEY = "analytics.data_model_relationships"
DATA_MODEL_USAGE_TABLE_KEY = "analytics.data_model_usage"
DATA_MODELS_TABLE_KEYS = (
    DATA_MODELS_TABLE_KEY,
    DATA_MODEL_FIELDS_TABLE_KEY,
    DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
)
DATA_MODELS_SAVE_CONTEXT = SaverContext(domain="analytics", target=DATA_MODELS_TARGET_NAME)
DATA_MODELS_CONTRACT = TableContractSpec(
    table_key=DATA_MODELS_TABLE_KEY,
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="data_models__base",
)
DATA_MODEL_FIELDS_CONTRACT = TableContractSpec(
    table_key=DATA_MODEL_FIELDS_TABLE_KEY,
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="data_model_fields__base",
)
DATA_MODEL_RELATIONSHIPS_CONTRACT = TableContractSpec(
    table_key=DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="data_model_relationships__base",
)
DATA_MODEL_USAGE_TARGET_NAME = "data_model_usage"
DATA_MODEL_USAGE_SAVE_CONTEXT = SaverContext(domain="analytics", target=DATA_MODEL_USAGE_TARGET_NAME)
DATA_MODEL_USAGE_CONTRACT = TableContractSpec(
    table_key=DATA_MODEL_USAGE_TABLE_KEY,
    domain="analytics",
    target=DATA_MODEL_USAGE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="data_model_usage__base",
)


def _module_map(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_map: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def data_models_result(env: BuildEnv, _q__core__modules: InferableTabularInput) -> DataModelsResult:
    inputs = load_data_models_inputs(env.gateway, env.snapshot)
    return compute_data_models_from_inputs(inputs, env.snapshot)


def data_models__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    return rows_to_frame(
        DATA_MODELS_TABLE_KEY,
        data_models_result.model_rows,
        columns=DATA_MODELS_COLS,
    )


@save_dataset(
    context=DATA_MODELS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=DATA_MODELS_TABLE_KEY),
)
@tag_dataset(domain="analytics", target=DATA_MODELS_TARGET_NAME, table_key=DATA_MODELS_TABLE_KEY)
@table_contract(DATA_MODELS_CONTRACT)
def data_models__table(data_models__base: pl.LazyFrame) -> pl.LazyFrame:
    return data_models__base


def data_model_fields__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    return rows_to_frame(
        DATA_MODEL_FIELDS_TABLE_KEY,
        data_models_result.field_rows,
        columns=DATA_MODEL_FIELDS_COLS,
    )


@save_dataset(
    context=DATA_MODELS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=DATA_MODEL_FIELDS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    table_key=DATA_MODEL_FIELDS_TABLE_KEY,
)
@table_contract(DATA_MODEL_FIELDS_CONTRACT)
def data_model_fields__table(data_model_fields__base: pl.LazyFrame) -> pl.LazyFrame:
    return data_model_fields__base


def data_model_relationships__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    return rows_to_frame(
        DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
        data_models_result.relationship_rows,
        columns=DATA_MODEL_RELATIONSHIPS_COLS,
    )


@save_dataset(
    context=DATA_MODELS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=DATA_MODEL_RELATIONSHIPS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    table_key=DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
)
@table_contract(DATA_MODEL_RELATIONSHIPS_CONTRACT)
def data_model_relationships__table(
    data_model_relationships__base: pl.LazyFrame,
) -> pl.LazyFrame:
    return data_model_relationships__base


data_models__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    table_keys=DATA_MODELS_TABLE_KEYS,
    node_name="data_models__table_materializations",
)


@codeintel_target(domain="analytics", target=DATA_MODELS_TARGET_NAME)
def t__data_models(
    env: BuildEnv,
    catalog: DagCatalog,
    data_models__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=DATA_MODELS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=data_models__table_materializations,
    )


def data_model_usage__base(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
    _q__core__goids: InferableTabularInput,
    _q__analytics__data_models: InferableTabularInput,
) -> pl.LazyFrame:
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    module_map = _module_map(modules_frame)
    if not module_map:
        return empty_frame_for_table(DATA_MODEL_USAGE_TABLE_KEY)
    catalog = CatalogService.from_db(env.gateway, repo=env.repo, commit=env.commit)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(env.gateway, request)
    return build_data_model_usage_rows(
        env.gateway,
        env.snapshot,
        module_map=module_map,
        ast_by_goid=ast_map,
        missing_goids=missing,
    )


@save_dataset(
    context=DATA_MODEL_USAGE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=DATA_MODEL_USAGE_TABLE_KEY),
)
@table_contract(DATA_MODEL_USAGE_CONTRACT)
def data_model_usage__table(data_model_usage__base: pl.LazyFrame) -> pl.LazyFrame:
    return data_model_usage__base


@codeintel_target(domain="analytics", target=DATA_MODEL_USAGE_TARGET_NAME)
def t__data_model_usage(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__data_model_usage: MaterializationResult,
) -> TargetRunRecord:
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=DATA_MODEL_USAGE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            DATA_MODEL_USAGE_TABLE_KEY: m__analytics__data_model_usage,
        },
    )


__all__ = [
    "data_model_fields__base",
    "data_model_fields__table",
    "data_model_relationships__base",
    "data_model_relationships__table",
    "data_model_usage__base",
    "data_model_usage__table",
    "data_models__base",
    "data_models__table",
    "data_models__table_materializations",
    "t__data_model_usage",
    "t__data_models",
]
