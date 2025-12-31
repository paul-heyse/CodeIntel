"""Coverage analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    SaverContext,
    save_relation_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, pl.LazyFrame)

COVERAGE_FUNCTIONS_TARGET_NAME = "coverage_functions"
COVERAGE_FUNCTIONS_TABLE_KEY = "analytics.coverage_functions"
COVERAGE_FUNCTIONS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
)


@save_relation_table(
    context=COVERAGE_FUNCTIONS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=COVERAGE_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=COVERAGE_FUNCTIONS_TARGET_NAME,
    table_key=COVERAGE_FUNCTIONS_TABLE_KEY,
)
def coverage_functions__table(env: BuildEnv) -> pl.LazyFrame:
    """Return an empty coverage functions frame.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame with the coverage functions schema.
    """
    _ = env
    return empty_frame_for_table(COVERAGE_FUNCTIONS_TABLE_KEY)


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
    "coverage_functions__table",
    "t__coverage_functions",
]
