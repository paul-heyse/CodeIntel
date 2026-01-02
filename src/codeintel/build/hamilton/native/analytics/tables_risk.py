"""Risk analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pandas as pd
import polars as pl
from hamilton.experimental.decorators.parameterize_frame import parameterize_frame

from codeintel.build.hamilton.column_ops import risk_features
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

RISK_FACTORS_TARGET_NAME = "risk_factors"
RISK_FACTORS_TABLE_KEY = "analytics.goid_risk_factors"
RISK_FACTORS_CONTRACT = TableContractSpec(
    table_key=RISK_FACTORS_TABLE_KEY,
    domain="analytics",
    target=RISK_FACTORS_TARGET_NAME,
    ops_module=risk_features,
    columns_to_pass=("risk_score", "cyclomatic_complexity"),
    required_cols=("risk_score",),
    clip_column=None,
    input_name="risk_factors__base",
)

RISK_LEVEL_HIGH_THRESHOLD = 5
RISK_LEVEL_MEDIUM_THRESHOLD = 3

_RISK_WEIGHT_SPEC = pd.DataFrame(
    [["risk_weight_low", "risk_weight_high", 0.3, 0.7]],
    columns=[
        ["out_low", "out_high", "low_weight", "high_weight"],
        ["out", "out", "value", "value"],
    ],
)


@parameterize_frame(_RISK_WEIGHT_SPEC)
def risk_weight_reference(low_weight: float, high_weight: float) -> pd.DataFrame:
    """Provide a minimal parameterized risk-weight lookup table.

    Returns
    -------
    pd.DataFrame
        Single-row lookup table with low/high risk weight columns.
    """
    return pd.DataFrame(
        {
            "risk_weight_low": [low_weight],
            "risk_weight_high": [high_weight],
        }
    )


def risk_factors__base(
    q__analytics__function_metrics: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build risk factors using function metrics and the call graph.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing risk factor rows.
    """
    frame = tabular_to_lazyframe(q__analytics__function_metrics)
    edges = tabular_to_lazyframe(q__graph__call_graph_edges)
    fan_in = (
        edges.group_by("callee_goid_h128")
        .len()
        .rename({"callee_goid_h128": "function_goid_h128", "len": "fan_in_count"})
    )
    fan_out = (
        edges.group_by("caller_goid_h128")
        .len()
        .rename({"caller_goid_h128": "function_goid_h128", "len": "fan_out_count"})
    )

    risk_score = pl.col("cyclomatic_complexity").fill_null(0).cast(pl.Int64)
    risk_level = (
        pl.when(risk_score >= RISK_LEVEL_HIGH_THRESHOLD)
        .then(pl.lit("high"))
        .when(risk_score >= RISK_LEVEL_MEDIUM_THRESHOLD)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("risk_level")
    )
    frame = (
        frame.join(fan_in, on="function_goid_h128", how="left")
        .join(fan_out, on="function_goid_h128", how="left")
        .with_columns(
            risk_score.alias("risk_score"),
            risk_level,
            pl.col("fan_in_count").fill_null(0).cast(pl.Int64),
            pl.col("fan_out_count").fill_null(0).cast(pl.Int64),
            pl.lit(value=False).alias("has_tests"),
        )
    )
    return frame.select(
        [
            "function_goid_h128",
            "repo",
            "commit",
            "risk_score",
            "risk_level",
            "cyclomatic_complexity",
            "fan_in_count",
            "fan_out_count",
            "has_tests",
        ]
    )


_MODULE = sys.modules[__name__]
_RISK_FACTORS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=RISK_FACTORS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=RISK_FACTORS_TABLE_KEY,
            base_node="risk_factors__base",
            contract=RISK_FACTORS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=RISK_FACTORS_TABLE_KEY),
            node_name="risk_factors__table",
        ),
    ),
    table_materializations_node="risk_factors__table_materializations",
    anchor_node_name="t__risk_factors",
)
attach_table_target_template(_MODULE, spec=_RISK_FACTORS_TABLE_TARGET_SPEC)
risk_factors__table = _MODULE.risk_factors__table
risk_factors__table_materializations = _MODULE.risk_factors__table_materializations
t__risk_factors = _MODULE.t__risk_factors


__all__ = [
    "risk_factors__base",
    "risk_factors__table",
    "t__risk_factors",
]
