"""Module analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.profiles.modules import (
    build_module_profile_rows,
    compute_module_profile_inputs,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import module_features
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
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

MODULE_PROFILE_TARGET_NAME = "module_profile"
MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
MODULE_PROFILE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=MODULE_PROFILE_TARGET_NAME,
)
MODULE_PROFILE_CONTRACT = TableContractSpec(
    table_key=MODULE_PROFILE_TABLE_KEY,
    domain="analytics",
    target=MODULE_PROFILE_TARGET_NAME,
    ops_module=module_features,
    columns_to_pass=("total_loc", "function_count", "avg_risk_score", "module_coverage_ratio"),
    required_cols=("total_loc", "function_count"),
    clip_column=None,
    input_name="module_profile__base",
)


def module_profile__base(
    env: BuildEnv,
    _q__core__modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build module profile rows using gateway-backed helpers.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing module profile rows.
    """
    inputs = compute_module_profile_inputs(env.gateway, env.snapshot)
    rows = list(build_module_profile_rows(inputs))
    return rows_to_frame(MODULE_PROFILE_TABLE_KEY, rows)


@save_dataset(
    context=MODULE_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=MODULE_PROFILE_TABLE_KEY),
)
@table_contract(MODULE_PROFILE_CONTRACT)
def module_profile__table(module_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched module profile frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched module profile frame.
    """
    return module_profile__base


@codeintel_target(domain="analytics", target=MODULE_PROFILE_TARGET_NAME)
def t__module_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__module_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize module_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the module_profile target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=MODULE_PROFILE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            MODULE_PROFILE_TABLE_KEY: m__analytics__module_profile,
        },
    )


__all__ = [
    "module_profile__base",
    "module_profile__table",
    "t__module_profile",
]
