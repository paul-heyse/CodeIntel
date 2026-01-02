"""Function typing tables built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
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
FUNCTION_TYPES_COLUMNS = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "total_params",
    "annotated_params",
    "unannotated_params",
    "param_typed_ratio",
    "has_return_annotation",
    "return_type",
    "return_type_source",
    "type_comment",
    "param_types",
    "fully_typed",
    "partial_typed",
    "untyped",
    "typedness_bucket",
    "typedness_source",
    "created_at",
)


def function_types__base(function_analytics_result: FunctionAnalyticsResult) -> pl.LazyFrame:
    """Build function typing rows from computed analytics.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function typing coverage columns.
    """
    return rows_to_frame(
        FUNCTION_TYPES_TABLE_KEY,
        function_analytics_result.types_rows,
        columns=FUNCTION_TYPES_COLUMNS,
    )


_MODULE = sys.modules[__name__]
_FUNCTION_TYPES_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_TYPES_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_TYPES_TABLE_KEY,
            base_node="function_types__base",
            contract=FUNCTION_TYPES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_TYPES_TABLE_KEY),
            node_name="function_types__table",
        ),
    ),
    table_materializations_node="function_types__table_materializations",
    anchor_node_name="t__function_types",
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
