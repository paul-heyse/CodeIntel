"""Function validation table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.build.analytics.parsing.compute import get_validation_rows
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.core.columnar.rows import empty_table_for_table

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, FunctionAnalyticsResult)

FUNCTION_VALIDATION_TARGET_NAME = "function_validation"
FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
FUNCTION_VALIDATION_CONTRACT = contract_ref_for_table(
    table_key=FUNCTION_VALIDATION_TABLE_KEY,
    target_name=FUNCTION_VALIDATION_TARGET_NAME,
    input_name="function_validation__base",
    required_cols=(),
    clip_column=None,
)


def function_validation__base(
    function_analytics_result: FunctionAnalyticsResult,
) -> pa.Table:
    """Build function validation rows from the analytics reporter.

    Returns
    -------
    pa.Table
        Reader with function validation rows.
    """
    rows = get_validation_rows(function_analytics_result.reporter, None).function_rows
    if not rows:
        return empty_table_for_table(FUNCTION_VALIDATION_TABLE_KEY)
    return finalize_analytics_rows(FUNCTION_VALIDATION_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_FUNCTION_VALIDATION_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=FUNCTION_VALIDATION_CONTRACT,
        input_type=pa.Table,
    )
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
