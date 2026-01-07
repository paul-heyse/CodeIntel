"""Function typing tables built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_TYPES_TARGET_NAME = "function_types"
FUNCTION_TYPES_TABLE_KEY = "analytics.function_types"
FUNCTION_TYPES_CONTRACT = TableContractSpec(
    table_key=FUNCTION_TYPES_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_TYPES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_types__base",
)


def function_types__base(
    function_analytics_result: FunctionAnalyticsResult,
) -> pa.Table:
    """Build function typing rows from computed analytics.

    Returns
    -------
    pa.Table
        Reader with function typing rows.
    """
    if not function_analytics_result.types_rows:
        return empty_table_for_table(FUNCTION_TYPES_TABLE_KEY)
    reader, _ = table_for_rows(
        FUNCTION_TYPES_TABLE_KEY,
        function_analytics_result.types_rows,
    )
    return reader


_MODULE = sys.modules[__name__]
_FUNCTION_TYPES_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=FUNCTION_TYPES_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_FUNCTION_TYPES_TABLE_TARGET_SPEC)
function_types__table = _MODULE.function_types__table
function_types__table_materializations = _MODULE.function_types__table_materializations
t__function_types = _MODULE.t__function_types


__all__ = [
    "function_types__base",
    "function_types__table",
    "t__function_types",
]
