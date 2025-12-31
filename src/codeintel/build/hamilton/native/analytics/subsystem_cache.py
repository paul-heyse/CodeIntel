"""Subsystem cache tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.subsystems.cache import (
    build_subsystem_coverage_cache_rows,
    build_subsystem_profile_cache_rows,
)
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
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_CACHES_TARGET_NAME = "subsystem_caches"
SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"
SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY = "analytics.subsystem_coverage_cache"
SUBSYSTEM_CACHE_TABLE_KEYS = (
    SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
    SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY,
)
SUBSYSTEM_CACHE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
)
SUBSYSTEM_PROFILE_CACHE_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_profile_cache__base",
)
SUBSYSTEM_COVERAGE_CACHE_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_coverage_cache__base",
)


def subsystem_profile_cache__base(
    env: BuildEnv,
    _q__analytics__subsystems: InferableTabularInput,
    _q__analytics__subsystem_graph_metrics: InferableTabularInput,
    _q__analytics__module_profile: InferableTabularInput,
    _q__analytics__entrypoints: InferableTabularInput,
) -> pl.LazyFrame:
    """Build cached subsystem profile rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem profile cache rows.
    """
    rows = build_subsystem_profile_cache_rows(env.gateway, env.snapshot)
    return rows_to_frame(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY, rows)


@save_dataset(
    context=SUBSYSTEM_CACHE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEM_PROFILE_CACHE_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    table_key=SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
)
@table_contract(SUBSYSTEM_PROFILE_CACHE_CONTRACT)
def subsystem_profile_cache__table(
    subsystem_profile_cache__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist subsystem profile cache rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem profile cache frame.
    """
    return subsystem_profile_cache__base


def subsystem_coverage_cache__base(
    env: BuildEnv,
    _q__analytics__subsystems: InferableTabularInput,
    _q__analytics__subsystem_modules: InferableTabularInput,
    _q__analytics__test_profile: InferableTabularInput,
    _q__analytics__coverage_functions: InferableTabularInput,
) -> pl.LazyFrame:
    """Build cached subsystem coverage rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem coverage cache rows.
    """
    rows = build_subsystem_coverage_cache_rows(env.gateway, env.snapshot)
    return rows_to_frame(SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY, rows)


@save_dataset(
    context=SUBSYSTEM_CACHE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    table_key=SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY,
)
@table_contract(SUBSYSTEM_COVERAGE_CACHE_CONTRACT)
def subsystem_coverage_cache__table(
    subsystem_coverage_cache__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist subsystem coverage cache rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem coverage cache frame.
    """
    return subsystem_coverage_cache__base


subsystem_caches__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    table_keys=SUBSYSTEM_CACHE_TABLE_KEYS,
    node_name="subsystem_caches__table_materializations",
)


@codeintel_target(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
def t__subsystem_caches(
    env: BuildEnv,
    catalog: DagCatalog,
    subsystem_caches__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize subsystem_caches target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the subsystem_caches target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SUBSYSTEM_CACHES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=subsystem_caches__table_materializations,
    )


__all__ = [
    "subsystem_caches__table_materializations",
    "subsystem_coverage_cache__base",
    "subsystem_coverage_cache__table",
    "subsystem_profile_cache__base",
    "subsystem_profile_cache__table",
    "t__subsystem_caches",
]
