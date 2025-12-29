"""Example workspace target using the CodeIntel SDK."""

from __future__ import annotations

import polars as pl

from codeintel.sdk import (
    BuildEnv,
    DagCatalog,
    MaterializationBundle,
    MaterializationResult,
    TableSaveOptions,
    TargetRunRecord,
    finalize_materializations,
    save_to_table,
    target_anchor,
)

TARGET_DOMAIN = "analytics"
TARGET_NAME = "hello_example"
TABLE_KEY = "analytics.hello_example"


@save_to_table(
    domain=TARGET_DOMAIN,
    target=TARGET_NAME,
    options=TableSaveOptions(table_key=TABLE_KEY),
)
def hello_example__rows() -> pl.LazyFrame:
    """Build a tiny example table as a LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing demo rows for the hello_example target.
    """
    frame = pl.DataFrame({"message": ["hello"], "value": [1]})
    return frame.lazy()


@target_anchor(domain=TARGET_DOMAIN, target=TARGET_NAME)
def t__hello_example(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__hello_example: MaterializationResult,
) -> TargetRunRecord:
    """Finalize the hello_example target from table materializations.

    Parameters
    ----------
    env
        Build environment for the target run.
    catalog
        DAG catalog for resolving metadata.
    m__analytics__hello_example
        Materialization result for the example table.

    Returns
    -------
    TargetRunRecord
        Run record describing the target materialization.
    """
    materializations = MaterializationBundle(
        table_materializations={TABLE_KEY: m__analytics__hello_example},
    )
    return finalize_materializations(
        env=env,
        catalog=catalog,
        target_name=TARGET_NAME,
        materializations=materializations,
    )


__all__ = [
    "hello_example__rows",
    "t__hello_example",
]
