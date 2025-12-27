"""Module analytics tables built with relation-first nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.column_ops import module_features
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

MODULE_PROFILE_TARGET_NAME = "module_profile"
MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
MODULE_PROFILE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=MODULE_PROFILE_TARGET_NAME,
)
MODULE_PROFILE_CONTRACT = TableContractSpec(
    table_key=MODULE_PROFILE_TABLE_KEY,
    domain="analytics",
    target=MODULE_PROFILE_TARGET_NAME,
    ops_module=module_features,
    columns_to_pass=("total_loc", "function_count", "avg_risk_score", "module_coverage_ratio"),
    required_cols=("total_loc", "function_count"),
    clip_column=None,
    input_name="module_profile__base",
)


def module_profile__base(q__core__modules: DuckDBRelation) -> pl.LazyFrame:
    """Build a minimal module profile frame from core.modules.

    Parameters
    ----------
    q__core__modules
        Relation for ``core.modules``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with module profile columns.
    """
    frame = relation_to_polars(q__core__modules, lazy=True)
    frame = frame.with_columns(
        pl.lit(1).cast(pl.Int64).alias("file_count"),
        pl.lit(0).cast(pl.Int64).alias("total_loc"),
        pl.lit(0).cast(pl.Int64).alias("function_count"),
        pl.lit(0).cast(pl.Int64).alias("class_count"),
        pl.lit(0).cast(pl.Float64).alias("avg_file_complexity"),
        pl.lit(0).cast(pl.Float64).alias("max_file_complexity"),
        pl.lit(0).cast(pl.Float64).alias("avg_risk_score"),
        pl.lit(0).cast(pl.Float64).alias("max_risk_score"),
        pl.lit(0).cast(pl.Float64).alias("module_coverage_ratio"),
        pl.lit(None).cast(pl.Datetime).alias("created_at"),
    )
    return frame.select(
        [
            "repo",
            "commit",
            "module",
            "path",
            "language",
            "file_count",
            "total_loc",
            "function_count",
            "class_count",
            "avg_file_complexity",
            "max_file_complexity",
            "avg_risk_score",
            "max_risk_score",
            "module_coverage_ratio",
            "tags",
            "owners",
            "created_at",
        ]
    )


@save_relation_table(
    context=MODULE_PROFILE_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=MODULE_PROFILE_TABLE_KEY),
)
@table_contract(MODULE_PROFILE_CONTRACT)
def module_profile__table(module_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched module profile frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched module profile frame.
    """
    return module_profile__base


@codeintel_target(domain="analytics", target=MODULE_PROFILE_TARGET_NAME)
def t__module_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__module_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize module_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the module_profile target.
    """
    return record_from_duckdb_materialization(
        env=env,
        catalog=catalog,
        target_name=MODULE_PROFILE_TARGET_NAME,
        expected_table_key=MODULE_PROFILE_TABLE_KEY,
        materialization=m__analytics__module_profile,
    )


__all__ = [
    "module_profile__base",
    "module_profile__table",
    "t__module_profile",
]
