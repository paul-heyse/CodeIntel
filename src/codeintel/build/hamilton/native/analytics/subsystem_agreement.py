"""Subsystem agreement checks built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.graphs.subsystem_agreement import build_subsystem_agreement_rows
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
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_AGREEMENT_TARGET_NAME = "subsystem_agreement"
SUBSYSTEM_AGREEMENT_TABLE_KEY = "analytics.subsystem_agreement"
SUBSYSTEM_AGREEMENT_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
)
SUBSYSTEM_AGREEMENT_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_agreement__base",
)

SUBSYSTEM_AGREEMENT_COLUMNS = (
    "repo",
    "commit",
    "module",
    "subsystem_id",
    "import_community_id",
    "agrees",
    "created_at",
)


def subsystem_agreement__base(
    env: BuildEnv,
    _q__analytics__subsystem_modules: InferableTabularInput,
    _q__analytics__graph_metrics_modules_ext: InferableTabularInput,
) -> pl.LazyFrame:
    """Build subsystem agreement rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem agreement rows.
    """
    rows = build_subsystem_agreement_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
    )
    return rows_to_frame(
        SUBSYSTEM_AGREEMENT_TABLE_KEY,
        rows,
        columns=SUBSYSTEM_AGREEMENT_COLUMNS,
    )


@save_dataset(
    context=SUBSYSTEM_AGREEMENT_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY,
)
@table_contract(SUBSYSTEM_AGREEMENT_CONTRACT)
def subsystem_agreement__table(
    subsystem_agreement__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist subsystem agreement rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem agreement frame.
    """
    return subsystem_agreement__base


@codeintel_target(domain="analytics", target=SUBSYSTEM_AGREEMENT_TARGET_NAME)
def t__subsystem_agreement(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__subsystem_agreement: MaterializationResult,
) -> TargetRunRecord:
    """Finalize subsystem_agreement target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the subsystem_agreement target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            SUBSYSTEM_AGREEMENT_TABLE_KEY: m__analytics__subsystem_agreement,
        },
    )


__all__ = [
    "subsystem_agreement__base",
    "subsystem_agreement__table",
    "t__subsystem_agreement",
]
