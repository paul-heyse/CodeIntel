"""Profile analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.analytics.hotspots import compute_hotspot_rows
from codeintel.build.analytics.profiles.files import (
    build_file_profile_rows,
    compute_file_profile_inputs,
)
from codeintel.build.analytics.profiles.functions import (
    FunctionProfileViews,
    build_function_profile_rows,
    compute_function_profile_inputs,
    join_function_contracts,
    join_function_coverage,
    join_function_docs,
    join_function_effects,
    join_function_risk,
    join_function_roles,
    load_function_base_info,
)
from codeintel.build.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.build.analytics.profiles.types import FileProfileInputs, FunctionProfileInputs
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_PROFILE_TARGET_NAME = "function_profile"
FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
FUNCTION_PROFILE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_PROFILE_TARGET_NAME,
)
FUNCTION_PROFILE_CONTRACT = TableContractSpec(
    table_key=FUNCTION_PROFILE_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_PROFILE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_profile__base",
)

FILE_PROFILE_TARGET_NAME = "file_profile"
FILE_PROFILE_TABLE_KEY = "analytics.file_profile"
FILE_PROFILE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FILE_PROFILE_TARGET_NAME,
)
FILE_PROFILE_CONTRACT = TableContractSpec(
    table_key=FILE_PROFILE_TABLE_KEY,
    domain="analytics",
    target=FILE_PROFILE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="file_profile__base",
)

HOTSPOTS_TARGET_NAME = "hotspots"
HOTSPOTS_TABLE_KEY = "analytics.hotspots"
HOTSPOTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=HOTSPOTS_TARGET_NAME,
)
HOTSPOTS_CONTRACT = TableContractSpec(
    table_key=HOTSPOTS_TABLE_KEY,
    domain="analytics",
    target=HOTSPOTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="hotspots__base",
)


def function_profile_inputs(
    env: BuildEnv,
    _q__analytics__function_metrics: InferableTabularInput,
    _q__analytics__function_types: InferableTabularInput,
    _q__analytics__goid_risk_factors: InferableTabularInput,
    _q__analytics__coverage_functions: InferableTabularInput,
    _q__analytics__graph_metrics_functions: InferableTabularInput,
    _q__analytics__function_effects: InferableTabularInput,
    _q__analytics__function_contracts: InferableTabularInput,
    _q__analytics__semantic_roles_functions: InferableTabularInput,
    _q__core__docstrings: InferableTabularInput,
    _q__analytics__typedness: InferableTabularInput,
    _q__analytics__static_diagnostics: InferableTabularInput,
) -> FunctionProfileInputs:
    return compute_function_profile_inputs(env.gateway, env.snapshot)


def function_profile__base(function_profile_inputs: FunctionProfileInputs) -> pl.LazyFrame:
    """Build function profile rows using gateway-backed helpers.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing function profile rows.
    """
    inputs = function_profile_inputs
    views = FunctionProfileViews(
        base_by_func=load_function_base_info(inputs),
        risk_by_func=join_function_risk(inputs),
        coverage_by_func=join_function_coverage(inputs),
        graph_by_func=summarize_graph_for_function_profile(inputs),
        effects_by_func=join_function_effects(inputs),
        contracts_by_func=join_function_contracts(inputs),
        roles_by_func=join_function_roles(inputs),
        docs_by_func=join_function_docs(inputs),
    )
    rows = list(build_function_profile_rows(inputs, views=views))
    return rows_to_frame(FUNCTION_PROFILE_TABLE_KEY, rows)


@save_dataset(
    context=FUNCTION_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_PROFILE_TABLE_KEY),
)
@table_contract(FUNCTION_PROFILE_CONTRACT)
def function_profile__table(function_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist function profile rows.

    Returns
    -------
    pl.LazyFrame
        Persisted function profile frame.
    """
    return function_profile__base


@codeintel_target(domain="analytics", target=FUNCTION_PROFILE_TARGET_NAME)
def t__function_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_profile target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_PROFILE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_PROFILE_TABLE_KEY: m__analytics__function_profile,
        },
    )


def file_profile_inputs(
    env: BuildEnv,
    _q__analytics__function_profile: InferableTabularInput,
    _q__core__ast_metrics: InferableTabularInput,
    _q__analytics__hotspots: InferableTabularInput,
    _q__analytics__typedness: InferableTabularInput,
    _q__analytics__static_diagnostics: InferableTabularInput,
    _q__core__modules: InferableTabularInput,
) -> FileProfileInputs:
    return compute_file_profile_inputs(env.gateway, env.snapshot)


def file_profile__base(file_profile_inputs: FileProfileInputs) -> pl.LazyFrame:
    """Build file profile rows using gateway-backed helpers.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing file profile rows.
    """
    inputs = file_profile_inputs
    rows = list(build_file_profile_rows(inputs))
    return rows_to_frame(FILE_PROFILE_TABLE_KEY, rows)


@save_dataset(
    context=FILE_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FILE_PROFILE_TABLE_KEY),
)
@table_contract(FILE_PROFILE_CONTRACT)
def file_profile__table(file_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist file profile rows.

    Returns
    -------
    pl.LazyFrame
        Persisted file profile frame.
    """
    return file_profile__base


@codeintel_target(domain="analytics", target=FILE_PROFILE_TARGET_NAME)
def t__file_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__file_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize file_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the file_profile target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FILE_PROFILE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FILE_PROFILE_TABLE_KEY: m__analytics__file_profile,
        },
    )


def hotspots__base(q__core__ast_metrics: InferableTabularInput) -> pl.LazyFrame:
    """Build hotspot rows from core AST metrics.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing hotspot rows.
    """
    metrics_frame = tabular_to_lazyframe(q__core__ast_metrics).collect()
    ast_metrics: list[tuple[str, float]] = []
    for row in metrics_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        complexity_raw = row.get("complexity")
        complexity = float(complexity_raw or 0.0)
        ast_metrics.append((rel_path, complexity))
    rows = compute_hotspot_rows(ast_metrics)
    return rows_to_frame(HOTSPOTS_TABLE_KEY, rows)


@save_dataset(
    context=HOTSPOTS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=HOTSPOTS_TABLE_KEY),
)
@table_contract(HOTSPOTS_CONTRACT)
def hotspots__table(hotspots__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist hotspot rows.

    Returns
    -------
    pl.LazyFrame
        Persisted hotspots frame.
    """
    return hotspots__base


@codeintel_target(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def t__hotspots(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__hotspots: MaterializationResult,
) -> TargetRunRecord:
    """Finalize hotspots target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the hotspots target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=HOTSPOTS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            HOTSPOTS_TABLE_KEY: m__analytics__hotspots,
        },
    )


__all__ = [
    "file_profile__base",
    "file_profile__table",
    "function_profile__base",
    "function_profile__table",
    "hotspots__base",
    "hotspots__table",
    "t__file_profile",
    "t__function_profile",
    "t__hotspots",
]
