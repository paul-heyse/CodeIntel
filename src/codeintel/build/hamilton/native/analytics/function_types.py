"""Function typing tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
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

FUNCTION_TYPES_TARGET_NAME = "function_types"
FUNCTION_TYPES_TABLE_KEY = "analytics.function_types"
FUNCTION_TYPES_SAVE_CONTEXT = SaverContext(domain="analytics", target=FUNCTION_TYPES_TARGET_NAME)
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


def function_types__base(q__core__goids: InferableTabularInput) -> pl.LazyFrame:
    """Build a minimal function types frame from core.goids.

    Parameters
    ----------
    q__core__goids
        Relation for ``core.goids``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function typing coverage columns.
    """
    frame = tabular_to_lazyframe(q__core__goids)
    frame = frame.rename({"goid_h128": "function_goid_h128"})
    frame = frame.with_columns(
        pl.lit(0).cast(pl.Int64).alias("total_params"),
        pl.lit(0).cast(pl.Int64).alias("annotated_params"),
        pl.lit(0).cast(pl.Int64).alias("unannotated_params"),
        pl.lit(0.0).cast(pl.Float64).alias("param_typed_ratio"),
        pl.lit(value=False).cast(pl.Boolean).alias("has_return_annotation"),
        pl.lit(None).cast(pl.Utf8).alias("return_type"),
        pl.lit(None).cast(pl.Utf8).alias("return_type_source"),
        pl.lit(None).cast(pl.Utf8).alias("type_comment"),
        pl.lit("[]").alias("param_types"),
        pl.lit(value=False).cast(pl.Boolean).alias("fully_typed"),
        pl.lit(value=False).cast(pl.Boolean).alias("partial_typed"),
        pl.lit(value=True).cast(pl.Boolean).alias("untyped"),
        pl.lit("untyped").alias("typedness_bucket"),
        pl.lit(None).cast(pl.Utf8).alias("typedness_source"),
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
        ]
    )


@save_dataset(
    context=FUNCTION_TYPES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_TYPES_TABLE_KEY),
)
@table_contract(FUNCTION_TYPES_CONTRACT)
def function_types__table(function_types__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function types frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function types frame.
    """
    return function_types__base


@codeintel_target(domain="analytics", target=FUNCTION_TYPES_TARGET_NAME)
def t__function_types(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_types: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_types target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_types target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_TYPES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_TYPES_TABLE_KEY: m__analytics__function_types,
        },
    )


__all__ = [
    "function_types__base",
    "function_types__table",
    "t__function_types",
]
