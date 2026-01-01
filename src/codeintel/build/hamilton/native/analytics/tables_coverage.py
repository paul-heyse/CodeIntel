"""Coverage analytics tables built with inferable tabular nodes."""

from __future__ import annotations

from datetime import UTC, datetime

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
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, pl.LazyFrame)

COVERAGE_FUNCTIONS_TARGET_NAME = "coverage_functions"
COVERAGE_FUNCTIONS_TABLE_KEY = "analytics.coverage_functions"
COVERAGE_FUNCTIONS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
)
COVERAGE_FUNCTIONS_CONTRACT = TableContractSpec(
    table_key=COVERAGE_FUNCTIONS_TABLE_KEY,
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="coverage_functions__base",
)


def coverage_functions__base(
    env: BuildEnv,
    _q__analytics__coverage_lines: InferableTabularInput,
    _q__core__goids: InferableTabularInput,
) -> pl.LazyFrame:
    """Build the coverage functions frame from coverage lines and GOIDs.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with the coverage functions schema.
    """
    goids = tabular_to_lazyframe(_q__core__goids)
    coverage = tabular_to_lazyframe(_q__analytics__coverage_lines)

    predicate = (pl.col("repo") == env.snapshot.repo) & (pl.col("commit") == env.snapshot.commit)
    goids = (
        goids.filter(predicate)
        .filter(pl.col("kind").is_in(["function", "method"]))
        .select(
            pl.col("goid_h128").alias("function_goid_h128"),
            "urn",
            "repo",
            "commit",
            "rel_path",
            "language",
            "kind",
            "qualname",
            "start_line",
            "end_line",
        )
    )
    coverage = coverage.filter(predicate).select(
        "repo",
        "commit",
        "rel_path",
        "line",
        "is_executable",
        "is_covered",
    )

    joined = goids.join(coverage, on=["repo", "commit", "rel_path"], how="left")
    bounded = joined.filter(
        pl.col("line").is_null()
        | (
            (pl.col("line") >= pl.col("start_line"))
            & (pl.col("line") <= pl.coalesce(pl.col("end_line"), pl.col("start_line")))
        )
    )

    aggregated = bounded.group_by(
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
        ]
    ).agg(
        pl.when(pl.col("is_executable").fill_null(value=False))
        .then(1)
        .otherwise(0)
        .sum()
        .alias("executable_lines_raw"),
        pl.when(
            pl.col("is_executable").fill_null(value=False)
            & pl.col("is_covered").fill_null(value=False)
        )
        .then(1)
        .otherwise(0)
        .sum()
        .alias("covered_lines_raw"),
    )

    executable = pl.col("executable_lines_raw").fill_null(0)
    covered = pl.col("covered_lines_raw").fill_null(0)
    coverage_ratio = pl.when(executable == 0).then(None).otherwise(covered / executable)
    untested_reason = (
        pl.when(executable == 0)
        .then(pl.lit("no_executable_code"))
        .when(covered == 0)
        .then(pl.lit("no_tests"))
        .otherwise(pl.lit(""))
    )
    return aggregated.with_columns(
        executable.alias("executable_lines"),
        covered.alias("covered_lines"),
        coverage_ratio.alias("coverage_ratio"),
        (covered > 0).alias("tested"),
        untested_reason.alias("untested_reason"),
        pl.lit(datetime.now(tz=UTC)).alias("created_at"),
    ).select(
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
        "executable_lines",
        "covered_lines",
        "coverage_ratio",
        "tested",
        "untested_reason",
        "created_at",
    )


@save_dataset(
    context=COVERAGE_FUNCTIONS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=COVERAGE_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
    table_key=COVERAGE_FUNCTIONS_TABLE_KEY,
)
@table_contract(COVERAGE_FUNCTIONS_CONTRACT)
def coverage_functions__table(coverage_functions__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched coverage functions frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched coverage functions frame.
    """
    return coverage_functions__base


@codeintel_target(domain="analytics", target=COVERAGE_FUNCTIONS_TARGET_NAME)
def t__coverage_functions(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__coverage_functions: MaterializationResult,
) -> TargetRunRecord:
    """Finalize coverage_functions target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the coverage_functions target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=COVERAGE_FUNCTIONS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            COVERAGE_FUNCTIONS_TABLE_KEY: m__analytics__coverage_functions,
        },
    )


__all__ = [
    "coverage_functions__base",
    "coverage_functions__table",
    "t__coverage_functions",
]
