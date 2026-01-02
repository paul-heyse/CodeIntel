"""Function analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl
from hamilton.function_modifiers import cache

from codeintel.build.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_analytics_result_from_tabular,
)
from codeintel.build.hamilton.column_ops import function_features
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_METRICS_TARGET_NAME = "function_metrics"
FUNCTION_METRICS_TABLE_KEY = "analytics.function_metrics"
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


@cache()
def function_analytics_result(
    env: BuildEnv, q__core__goids: InferableTabularInput
) -> FunctionAnalyticsResult:
    """Compute function analytics rows from core.goids.

    Returns
    -------
    FunctionAnalyticsResult
        Metrics/types rows plus validation reporter.
    """
    return compute_function_analytics_result_from_tabular(q__core__goids, env.snapshot)


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


_MODULE = sys.modules[__name__]
_FUNCTION_METRICS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_METRICS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_METRICS_TABLE_KEY,
            base_node="function_metrics__base",
            contract=FUNCTION_METRICS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_METRICS_TABLE_KEY),
            node_name="function_metrics__table",
        ),
    ),
    table_materializations_node="function_metrics__table_materializations",
    anchor_node_name="t__function_metrics",
)
attach_table_target_template(_MODULE, spec=_FUNCTION_METRICS_TABLE_TARGET_SPEC)
function_metrics__table = _MODULE.function_metrics__table
function_metrics__table_materializations = _MODULE.function_metrics__table_materializations
t__function_metrics = _MODULE.t__function_metrics


__all__ = [
    "function_analytics_result",
    "function_metrics__base",
    "function_metrics__table",
    "t__function_metrics",
]
