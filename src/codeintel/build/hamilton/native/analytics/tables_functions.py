"""Function analytics tables built with relation-first nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import function_features
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.duckdb_relation import relation_to_polars
from codeintel.build.tabular.types import TabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, TabularInput)

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


def function_metrics__base(q__core__goids: TabularInput) -> pl.LazyFrame:
    """Build a minimal function metrics frame from core.goids.

    Parameters
    ----------
    q__core__goids
        Relation for ``core.goids``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function metrics columns.
    """
    frame = relation_to_polars(q__core__goids, lazy=True)
    end_line = pl.coalesce([pl.col("end_line"), pl.col("start_line")])
    frame = frame.rename({"goid_h128": "function_goid_h128"})
    frame = frame.with_columns(
        end_line.alias("end_line"),
        (end_line - pl.col("start_line") + 1).clip(lower_bound=0).alias("loc"),
        pl.lit(0).cast(pl.Int64).alias("cyclomatic_complexity"),
    )
    return frame.select(
        [
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
            "loc",
            "cyclomatic_complexity",
            "created_at",
        ]
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
    return record_from_duckdb_materialization(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_METRICS_TARGET_NAME,
        expected_table_key=FUNCTION_METRICS_TABLE_KEY,
        materialization=m__analytics__function_metrics,
    )


__all__ = [
    "function_metrics__base",
    "function_metrics__table",
    "t__function_metrics",
]
