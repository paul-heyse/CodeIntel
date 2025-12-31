"""History timeseries aggregation target."""

from __future__ import annotations

import logging

import polars as pl

from codeintel.analytics.history.history_timeseries import (
    HISTORY_TIMESERIES_TABLE_KEY,
    build_history_timeseries_rows,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.ingestion.frame_utils import empty_lazyframe_for_table
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

LOG = logging.getLogger(__name__)

HISTORY_TIMESERIES_TARGET_NAME = "history_timeseries"
HISTORY_TIMESERIES_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
)
HISTORY_TIMESERIES_CONTRACT = TableContractSpec(
    table_key=HISTORY_TIMESERIES_TABLE_KEY,
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="history_timeseries__base",
)


def history_timeseries__base(env: BuildEnv) -> pl.LazyFrame:
    """Build a frame for analytics.history_timeseries.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for analytics.history_timeseries.
    """
    options = env.history_options
    resolver = env.history_db_resolver
    if options is None or resolver is None:
        LOG.info("history_timeseries skipped: history options or resolver not provided.")
        return empty_lazyframe_for_table(HISTORY_TIMESERIES_TABLE_KEY)
    return build_history_timeseries_rows(
        snapshot=env.snapshot,
        gateway_resolver=resolver,
        options=options,
    )


@save_dataset(
    context=HISTORY_TIMESERIES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=HISTORY_TIMESERIES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
    table_key=HISTORY_TIMESERIES_TABLE_KEY,
)
@table_contract(HISTORY_TIMESERIES_CONTRACT)
def history_timeseries__table(history_timeseries__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist history timeseries rows.

    Returns
    -------
    pl.LazyFrame
        Persisted history timeseries frame.
    """
    return history_timeseries__base


@codeintel_target(domain="analytics", target=HISTORY_TIMESERIES_TARGET_NAME)
def t__history_timeseries(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__history_timeseries: MaterializationResult,
) -> TargetRunRecord:
    """Finalize history_timeseries target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the history_timeseries target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=HISTORY_TIMESERIES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            HISTORY_TIMESERIES_TABLE_KEY: m__analytics__history_timeseries,
        },
    )


__all__ = [
    "HISTORY_TIMESERIES_TARGET_NAME",
    "history_timeseries__base",
    "history_timeseries__table",
    "t__history_timeseries",
]
