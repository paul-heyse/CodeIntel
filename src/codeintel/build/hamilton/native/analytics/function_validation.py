"""Function validation table built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.build.analytics.parsing.compute import get_validation_rows
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

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, FunctionAnalyticsResult)

FUNCTION_VALIDATION_TARGET_NAME = "function_validation"
FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
FUNCTION_VALIDATION_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_VALIDATION_TARGET_NAME,
)
FUNCTION_VALIDATION_CONTRACT = TableContractSpec(
    table_key=FUNCTION_VALIDATION_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_VALIDATION_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_validation__base",
)


def function_validation__base(
    function_analytics_result: FunctionAnalyticsResult,
) -> pl.LazyFrame:
    """Build function validation rows from the analytics reporter.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function validation columns.
    """
    rows = get_validation_rows(function_analytics_result.reporter, None).function_rows
    return rows_to_frame(FUNCTION_VALIDATION_TABLE_KEY, rows)


@save_dataset(
    context=FUNCTION_VALIDATION_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_VALIDATION_TABLE_KEY),
)
@table_contract(FUNCTION_VALIDATION_CONTRACT)
def function_validation__table(function_validation__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function validation frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function validation frame.
    """
    return function_validation__base


@codeintel_target(domain="analytics", target=FUNCTION_VALIDATION_TARGET_NAME)
def t__function_validation(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_validation: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_validation target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_validation target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_VALIDATION_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_VALIDATION_TABLE_KEY: m__analytics__function_validation,
        },
    )


__all__ = [
    "function_validation__base",
    "function_validation__table",
    "t__function_validation",
]
