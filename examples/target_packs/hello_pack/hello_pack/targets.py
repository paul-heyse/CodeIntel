"""Example target module for the hello pack."""

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
TARGET_NAME = "hello_pack_example"
TABLE_KEY = "analytics.hello_pack_example"


@save_to_table(
    domain=TARGET_DOMAIN,
    target=TARGET_NAME,
    options=TableSaveOptions(table_key=TABLE_KEY),
)
def hello_pack_example__rows() -> pl.LazyFrame:
    """Build a tiny example table as a LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with the example data.
    """
    frame = pl.DataFrame({"message": ["hello pack"], "value": [1]})
    return frame.lazy()


@target_anchor(domain=TARGET_DOMAIN, target=TARGET_NAME)
def t__hello_pack_example(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__hello_pack_example: MaterializationResult,
) -> TargetRunRecord:
    """Finalize the hello_pack_example target from table materializations.

    Parameters
    ----------
    env
        Build environment for the target run.
    catalog
        DAG catalog for resolving metadata.
    m__analytics__hello_pack_example
        Materialization result for the example table.

    Returns
    -------
    TargetRunRecord
        Materialization record for the target.
    """
    return finalize_materializations(
        env=env,
        catalog=catalog,
        target_name=TARGET_NAME,
        materializations=MaterializationBundle(
            table_materializations={TABLE_KEY: m__analytics__hello_pack_example}
        ),
    )


__all__ = [
    "hello_pack_example__rows",
    "t__hello_pack_example",
]
