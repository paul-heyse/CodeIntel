"""Function analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_analytics_result_from_table,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import function_features
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

FUNCTION_METRICS_TARGET_NAME = "function_metrics"
FUNCTION_METRICS_TABLE_KEY = "analytics.function_metrics"
FUNCTION_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
)
FUNCTION_METRICS_CONTRACT = TableContractSpec(
    table_key=FUNCTION_METRICS_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    ops_module=function_features,
    columns_to_pass=("loc", "cyclomatic_complexity"),
    required_cols=("loc",),
    clip_column="loc",
    input_name="function_metrics__base",
)


def function_analytics_result(
    env: BuildEnv, _q__core__goids: InferableTabularInput
) -> FunctionAnalyticsResult:
    """Compute function analytics rows from core.goids.

    Returns
    -------
    FunctionAnalyticsResult
        Metrics/types rows plus validation reporter.
    """
    relation = env.gateway.relation_from_table_key("core.goids")
    return compute_function_analytics_result_from_table(relation, env.snapshot)


def function_metrics__base(function_analytics_result: FunctionAnalyticsResult) -> pl.LazyFrame:
    """Build function metrics rows from computed analytics.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function metrics columns.
    """
    return rows_to_frame(
        FUNCTION_METRICS_TABLE_KEY,
        function_analytics_result.metrics_rows,
    )


@save_dataset(
    context=FUNCTION_METRICS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_METRICS_TABLE_KEY),
)
@table_contract(FUNCTION_METRICS_CONTRACT)
def function_metrics__table(function_metrics__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function metrics frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function metrics frame.
    """
    return function_metrics__base


@codeintel_target(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def t__function_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_metrics: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_METRICS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_METRICS_TABLE_KEY: m__analytics__function_metrics,
        },
    )


__all__ = [
    "function_analytics_result",
    "function_metrics__base",
    "function_metrics__table",
    "t__function_metrics",
]
