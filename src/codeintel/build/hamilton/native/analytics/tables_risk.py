"""Risk analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import risk_features
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

RISK_FACTORS_TARGET_NAME = "risk_factors"
RISK_FACTORS_TABLE_KEY = "analytics.goid_risk_factors"
RISK_FACTORS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=RISK_FACTORS_TARGET_NAME,
)
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


def risk_factors__base(
    q__analytics__function_metrics: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__analytics__test_coverage_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build risk factors using function metrics, call graph, and test coverage.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing risk factor rows.
    """
    frame = tabular_to_lazyframe(q__analytics__function_metrics)
    edges = tabular_to_lazyframe(q__graph__call_graph_edges)
    coverage = tabular_to_lazyframe(q__analytics__test_coverage_edges)

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
    tested = (
        coverage.group_by("function_goid_h128")
        .len()
        .with_columns(pl.lit(value=True).alias("has_tests"))
        .select(["function_goid_h128", "has_tests"])
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
        .join(tested, on="function_goid_h128", how="left")
        .with_columns(
            risk_score.alias("risk_score"),
            risk_level,
            pl.col("fan_in_count").fill_null(0).cast(pl.Int64),
            pl.col("fan_out_count").fill_null(0).cast(pl.Int64),
            pl.col("has_tests").fill_null(value=False).cast(pl.Boolean),
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


@save_dataset(
    context=RISK_FACTORS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=RISK_FACTORS_TABLE_KEY),
)
@table_contract(RISK_FACTORS_CONTRACT)
def risk_factors__table(risk_factors__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched risk factors frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched risk factors frame.
    """
    return risk_factors__base


@codeintel_target(domain="analytics", target=RISK_FACTORS_TARGET_NAME)
def t__risk_factors(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__goid_risk_factors: MaterializationResult,
) -> TargetRunRecord:
    """Finalize risk_factors target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the risk_factors target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=RISK_FACTORS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            RISK_FACTORS_TABLE_KEY: m__analytics__goid_risk_factors,
        },
    )


__all__ = [
    "risk_factors__base",
    "risk_factors__table",
    "t__risk_factors",
]
