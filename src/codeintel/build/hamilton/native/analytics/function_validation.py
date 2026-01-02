"""Function validation table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.build.analytics.parsing.compute import get_validation_rows
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.frames import rows_to_frame
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, FunctionAnalyticsResult)

FUNCTION_VALIDATION_TARGET_NAME = "function_validation"
FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
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


_MODULE = sys.modules[__name__]
_FUNCTION_VALIDATION_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_VALIDATION_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_VALIDATION_TABLE_KEY,
            base_node="function_validation__base",
            contract=FUNCTION_VALIDATION_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_VALIDATION_TABLE_KEY),
            node_name="function_validation__table",
        ),
    ),
    table_materializations_node="function_validation__table_materializations",
    anchor_node_name="t__function_validation",
)
attach_table_target_template(_MODULE, spec=_FUNCTION_VALIDATION_TABLE_TARGET_SPEC)
function_validation__table = _MODULE.function_validation__table
function_validation__table_materializations = _MODULE.function_validation__table_materializations
t__function_validation = _MODULE.t__function_validation


__all__ = [
    "function_validation__base",
    "function_validation__table",
    "t__function_validation",
]
