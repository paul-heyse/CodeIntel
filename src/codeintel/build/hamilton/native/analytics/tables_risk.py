"""Risk analytics tables built with relation-first nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import risk_features
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    SaverContext,
    save_relation_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.duckdb_relation import relation_to_polars
from codeintel.storage.gateway import DuckDBRelation

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, DuckDBRelation)

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


def risk_factors__base(q__analytics__function_metrics: DuckDBRelation) -> pl.LazyFrame:
    """Build a minimal risk factors frame from function metrics.

    Parameters
    ----------
    q__analytics__function_metrics
        Relation for ``analytics.function_metrics``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with risk factor columns.
    """
    frame = relation_to_polars(q__analytics__function_metrics, lazy=True)
    risk_score = pl.col("cyclomatic_complexity").fill_null(0).cast(pl.Int64)
    risk_level = (
        pl.when(risk_score >= RISK_LEVEL_HIGH_THRESHOLD)
        .then(pl.lit("high"))
        .when(risk_score >= RISK_LEVEL_MEDIUM_THRESHOLD)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("risk_level")
    )
    frame = frame.with_columns(
        risk_score.alias("risk_score"),
        risk_level,
        pl.lit(0).cast(pl.Int64).alias("fan_in_count"),
        pl.lit(0).cast(pl.Int64).alias("fan_out_count"),
        pl.lit(value=False).alias("has_tests"),
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


@save_relation_table(
    context=RISK_FACTORS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=RISK_FACTORS_TABLE_KEY),
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
    return record_from_duckdb_materialization(
        env=env,
        catalog=catalog,
        target_name=RISK_FACTORS_TARGET_NAME,
        expected_table_key=RISK_FACTORS_TABLE_KEY,
        materialization=m__analytics__goid_risk_factors,
    )


__all__ = [
    "risk_factors__base",
    "risk_factors__table",
    "t__risk_factors",
]
