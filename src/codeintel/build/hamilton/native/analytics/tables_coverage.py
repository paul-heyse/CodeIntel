"""Coverage analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.compute.coverage import build_coverage_functions_expr
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
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
from codeintel.build.tabular.conversion import relation_to_polars_lazy
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
    relation = build_coverage_functions_expr(env.gateway, env.snapshot)
    if relation is None:
        return empty_frame_for_table(COVERAGE_FUNCTIONS_TABLE_KEY)
    return relation_to_polars_lazy(relation)


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
