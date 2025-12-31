"""Subsystem analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.subsystems.materialize import SubsystemRows, build_subsystem_rows
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

SUBSYSTEMS_TARGET_NAME = "subsystems"
SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_TABLE_KEYS = (SUBSYSTEMS_TABLE_KEY, SUBSYSTEM_MODULES_TABLE_KEY)
SUBSYSTEMS_SAVE_CONTEXT = SaverContext(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
SUBSYSTEMS_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEMS_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystems__base",
)
SUBSYSTEM_MODULES_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_MODULES_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_modules__base",
)

SUBSYSTEMS_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "modules_json",
    "entrypoints_json",
    "internal_edge_count",
    "external_edge_count",
    "fan_in",
    "fan_out",
    "function_count",
    "avg_risk_score",
    "max_risk_score",
    "high_risk_function_count",
    "risk_level",
    "created_at",
)

SUBSYSTEM_MODULES_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "module",
    "role",
)


def subsystem_rows(
    env: BuildEnv,
    _q__core__modules: InferableTabularInput,
    _q__graph__import_graph_edges: InferableTabularInput,
    _q__graph__symbol_use_edges: InferableTabularInput,
    _q__analytics__config_values: InferableTabularInput,
    _q__analytics__goid_risk_factors: InferableTabularInput,
    _q__analytics__function_metrics: InferableTabularInput,
) -> SubsystemRows:
    """Compute subsystem inference rows for subsystems and memberships.

    Returns
    -------
    SubsystemRows
        Subsystem summary and membership rows.
    """
    return build_subsystem_rows(env.gateway, env.snapshot)


def subsystems__base(subsystem_rows: SubsystemRows) -> pl.LazyFrame:
    """Build subsystem summary rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem rows.
    """
    return rows_to_frame(
        SUBSYSTEMS_TABLE_KEY,
        subsystem_rows.subsystem_rows,
        columns=SUBSYSTEMS_COLUMNS,
    )


@save_dataset(
    context=SUBSYSTEMS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEMS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    table_key=SUBSYSTEMS_TABLE_KEY,
)
@table_contract(SUBSYSTEMS_CONTRACT)
def subsystems__table(subsystems__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist subsystem summary rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem summary frame.
    """
    return subsystems__base


def subsystem_modules__base(subsystem_rows: SubsystemRows) -> pl.LazyFrame:
    """Build subsystem membership rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem membership rows.
    """
    return rows_to_frame(
        SUBSYSTEM_MODULES_TABLE_KEY,
        subsystem_rows.membership_rows,
        columns=SUBSYSTEM_MODULES_COLUMNS,
    )


@save_dataset(
    context=SUBSYSTEMS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEM_MODULES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    table_key=SUBSYSTEM_MODULES_TABLE_KEY,
)
@table_contract(SUBSYSTEM_MODULES_CONTRACT)
def subsystem_modules__table(subsystem_modules__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist subsystem membership rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem membership frame.
    """
    return subsystem_modules__base


subsystems__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    table_keys=SUBSYSTEMS_TABLE_KEYS,
    node_name="subsystems__table_materializations",
)


@codeintel_target(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems(
    env: BuildEnv,
    catalog: DagCatalog,
    subsystems__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize subsystems target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the subsystems target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SUBSYSTEMS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=subsystems__table_materializations,
    )


__all__ = [
    "subsystem_modules__base",
    "subsystem_modules__table",
    "subsystem_rows",
    "subsystems__base",
    "subsystems__table",
    "subsystems__table_materializations",
    "t__subsystems",
]
