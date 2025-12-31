"""Function history table built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_TABLE_KEY,
    build_function_history_rows,
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
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_HISTORY_TARGET_NAME = "function_history"
FUNCTION_HISTORY_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
)
FUNCTION_HISTORY_CONTRACT = TableContractSpec(
    table_key=FUNCTION_HISTORY_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_history__base",
)


def function_history__base(
    env: BuildEnv,
    _q__analytics__function_metrics: InferableTabularInput,
    _q__core__modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build function history rows from git history.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function history columns.
    """
    runner = env.providers.tool_runner if env.providers else None
    rows = build_function_history_rows(env.gateway, env.snapshot, runner=runner)
    return rows_to_frame(FUNCTION_HISTORY_TABLE_KEY, rows)


@save_dataset(
    context=FUNCTION_HISTORY_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_HISTORY_TABLE_KEY),
)
@table_contract(FUNCTION_HISTORY_CONTRACT)
def function_history__table(function_history__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function history frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function history frame.
    """
    return function_history__base


@codeintel_target(domain="analytics", target=FUNCTION_HISTORY_TARGET_NAME)
def t__function_history(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_history: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_history target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_history target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_HISTORY_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_HISTORY_TABLE_KEY: m__analytics__function_history,
        },
    )


__all__ = [
    "function_history__base",
    "function_history__table",
    "t__function_history",
]
