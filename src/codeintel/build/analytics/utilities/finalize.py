"""Finalize helpers for analytics tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.build.tabular.finalize_ops import (
    FinalizeResult,
    finalize_reader,
    finalize_spec_for_table,
    finalize_table,
)
from codeintel.core.columnar.rows import ColumnarRowBuffer, table_for_rows

RowInput = ColumnarRowBuffer | Iterable[Mapping[str, object]] | Iterable[Sequence[object]]


def finalize_analytics_result(table_key: str, table: pa.Table) -> FinalizeResult:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    FinalizeResult
        Finalization result containing good and rejected rows.
    """
    return finalize_table(
        table,
        spec=finalize_spec_for_table(
            table_key,
            mode="tolerant",
            emit_artifacts=True,
        ),
    )


def finalize_analytics_reader(table_key: str, reader: pa.RecordBatchReader) -> FinalizeResult:
    """Finalize an analytics reader against its contract in tolerant mode.

    Returns
    -------
    FinalizeResult
        Finalization result containing good and rejected rows.
    """
    return finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            table_key,
            mode="tolerant",
            emit_artifacts=True,
        ),
    )


def finalize_analytics_table(table_key: str, table: pa.Table) -> pa.Table:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table.
    """
    return finalize_analytics_result(table_key, table).good


def finalize_analytics_rows(table_key: str, rows: RowInput) -> pa.Table:
    """Build and finalize an analytics table from row payloads.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table built from row inputs.
    """
    if isinstance(rows, ColumnarRowBuffer):
        table = rows.to_table()
    else:
        table, _ = table_for_rows(table_key, rows)
    return finalize_analytics_table(table_key, table)


def finalize_artifact_table_key(table_key: str, artifact: str) -> str:
    """Return the companion dataset key for finalize artifacts.

    Returns
    -------
    str
        Artifact table key with the artifact suffix applied.
    """
    return f"{table_key}__{artifact}"


def finalize_artifact_counts(result: FinalizeResult) -> dict[str, int]:
    """Return row counts for finalize artifacts.

    Returns
    -------
    dict[str, int]
        Row counts for errors, alignment, and stats artifacts.
    """
    return {
        "errors": result.errors.num_rows,
        "alignment": result.alignment.num_rows,
        "stats": result.stats.num_rows,
    }


__all__ = [
    "finalize_analytics_reader",
    "finalize_analytics_result",
    "finalize_analytics_rows",
    "finalize_analytics_table",
    "finalize_artifact_counts",
    "finalize_artifact_table_key",
]
