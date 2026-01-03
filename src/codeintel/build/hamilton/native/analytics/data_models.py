"""Data model analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import polars as pl
from hamilton.function_modifiers import cache

from codeintel.build.analytics.compute.data_models.usage import (
    DataModelUsageInputs,
    build_data_model_usage_rows,
)
from codeintel.build.analytics.data_models.compute import (
    DataModelsResult,
    compute_data_models_pure,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
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
from codeintel.build.tabular.conversion import tabular_to_frame
from codeintel.build.tabular.frames import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

DATA_MODELS_TARGET_NAME = "data_models"
DATA_MODELS_TABLE_KEY = "analytics.data_models"
DATA_MODEL_FIELDS_TABLE_KEY = "analytics.data_model_fields"
DATA_MODEL_RELATIONSHIPS_TABLE_KEY = "analytics.data_model_relationships"
DATA_MODEL_USAGE_TABLE_KEY = "analytics.data_model_usage"
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


@dataclass(frozen=True)
class DataModelUsageCoreFrames:
    modules_frame: pl.DataFrame
    goids_frame: pl.DataFrame
    data_models_frame: pl.DataFrame


@dataclass(frozen=True)
class DataModelUsageSubsystemFrames:
    subsystem_modules_frame: pl.DataFrame
    subsystems_frame: pl.DataFrame
    function_types_frame: pl.DataFrame


@dataclass(frozen=True)
class DataModelUsageFrames:
    core: DataModelUsageCoreFrames
    subsystems: DataModelUsageSubsystemFrames


def data_model_usage_core_frames(
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__analytics__data_models: InferableTabularInput,
) -> DataModelUsageCoreFrames:
    return DataModelUsageCoreFrames(
        modules_frame=tabular_to_frame(q__core__modules),
        goids_frame=tabular_to_frame(q__core__goids),
        data_models_frame=tabular_to_frame(q__analytics__data_models),
    )


def data_model_usage_subsystem_frames(
    q__analytics__subsystem_modules: InferableTabularInput,
    q__analytics__subsystems: InferableTabularInput,
    q__analytics__function_types: InferableTabularInput,
) -> DataModelUsageSubsystemFrames:
    return DataModelUsageSubsystemFrames(
        subsystem_modules_frame=tabular_to_frame(q__analytics__subsystem_modules),
        subsystems_frame=tabular_to_frame(q__analytics__subsystems),
        function_types_frame=tabular_to_frame(q__analytics__function_types),
    )


def data_model_usage_frames(
    data_model_usage_core_frames: DataModelUsageCoreFrames,
    data_model_usage_subsystem_frames: DataModelUsageSubsystemFrames,
) -> DataModelUsageFrames:
    return DataModelUsageFrames(
        core=data_model_usage_core_frames,
        subsystems=data_model_usage_subsystem_frames,
    )


def _module_map(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_map: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


@cache()
def data_models_result(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__core__docstrings: InferableTabularInput,
) -> DataModelsResult:
    """Compute data model rows from tabular inputs.

    Returns
    -------
    DataModelsResult
        Computed data model rows and metadata.
    """
    goids_frame = tabular_to_frame(q__core__goids)
    modules_frame = tabular_to_frame(q__core__modules)
    docstrings_frame = tabular_to_frame(q__core__docstrings)
    return compute_data_models_pure(
        env.snapshot,
        goids_frame=goids_frame,
        modules_frame=modules_frame,
        docstrings_frame=docstrings_frame,
    )


def data_models__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    """Build data model summary rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing data model rows.
    """
    return rows_to_frame(
        DATA_MODELS_TABLE_KEY,
        data_models_result.model_rows,
    )


def data_model_fields__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    """Build data model field rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing data model field rows.
    """
    return rows_to_frame(
        DATA_MODEL_FIELDS_TABLE_KEY,
        data_models_result.field_rows,
    )


def data_model_relationships__base(data_models_result: DataModelsResult) -> pl.LazyFrame:
    """Build data model relationship rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing data model relationship rows.
    """
    return rows_to_frame(
        DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
        data_models_result.relationship_rows,
    )


_MODULE = sys.modules[__name__]
_DATA_MODELS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=DATA_MODELS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=DATA_MODELS_TABLE_KEY,
            base_node="data_models__base",
            contract=DATA_MODELS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=DATA_MODELS_TABLE_KEY),
            node_name="data_models__table",
        ),
        TableTargetTableSpec(
            table_key=DATA_MODEL_FIELDS_TABLE_KEY,
            base_node="data_model_fields__base",
            contract=DATA_MODEL_FIELDS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=DATA_MODEL_FIELDS_TABLE_KEY),
            node_name="data_model_fields__table",
        ),
        TableTargetTableSpec(
            table_key=DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
            base_node="data_model_relationships__base",
            contract=DATA_MODEL_RELATIONSHIPS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=DATA_MODEL_RELATIONSHIPS_TABLE_KEY),
            node_name="data_model_relationships__table",
        ),
    ),
    table_materializations_node="data_models__table_materializations",
    anchor_node_name="t__data_models",
)
attach_table_target_template(_MODULE, spec=_DATA_MODELS_TABLE_TARGET_SPEC)
data_models__table = _MODULE.data_models__table
data_model_fields__table = _MODULE.data_model_fields__table
data_model_relationships__table = _MODULE.data_model_relationships__table
data_models__table_materializations = _MODULE.data_models__table_materializations
t__data_models = _MODULE.t__data_models


def data_model_usage__base(
    env: BuildEnv,
    data_model_usage_frames: DataModelUsageFrames,
) -> pl.LazyFrame:
    """Build data model usage rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing data model usage rows.
    """
    modules_frame = data_model_usage_frames.core.modules_frame
    goids_frame = data_model_usage_frames.core.goids_frame
    data_models_frame = data_model_usage_frames.core.data_models_frame
    subsystem_modules_frame = data_model_usage_frames.subsystems.subsystem_modules_frame
    subsystems_frame = data_model_usage_frames.subsystems.subsystems_frame
    function_types_frame = data_model_usage_frames.subsystems.function_types_frame
    module_map = _module_map(modules_frame)
    if not module_map:
        return empty_frame_for_table(DATA_MODEL_USAGE_TABLE_KEY)
    catalog = catalog_provider_from_frames(goids_frame=goids_frame, modules_frame=modules_frame)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(request)
    return build_data_model_usage_rows(
        env.snapshot,
        DataModelUsageInputs(
            module_map=module_map,
            ast_by_goid=ast_map,
            models_frame=data_models_frame,
            subsystem_modules_frame=subsystem_modules_frame,
            subsystems_frame=subsystems_frame,
            function_types_frame=function_types_frame,
            missing_goids=missing,
        ),
    )


_DATA_MODEL_USAGE_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=DATA_MODEL_USAGE_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=DATA_MODEL_USAGE_TABLE_KEY,
            base_node="data_model_usage__base",
            contract=DATA_MODEL_USAGE_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=DATA_MODEL_USAGE_TABLE_KEY),
            node_name="data_model_usage__table",
        ),
    ),
    table_materializations_node="data_model_usage__table_materializations",
    anchor_node_name="t__data_model_usage",
)
attach_table_target_template(_MODULE, spec=_DATA_MODEL_USAGE_TABLE_TARGET_SPEC)
data_model_usage__table = _MODULE.data_model_usage__table
data_model_usage__table_materializations = _MODULE.data_model_usage__table_materializations
t__data_model_usage = _MODULE.t__data_model_usage


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
