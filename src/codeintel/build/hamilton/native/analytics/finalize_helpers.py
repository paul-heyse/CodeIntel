"""Finalize helpers for analytics tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.rows import table_for_rows

RowInput = Iterable[Mapping[str, object]] | Iterable[Sequence[object]]


def finalize_analytics_table(table_key: str, table: pa.Table) -> pa.Table:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table.
    """
    result = finalize_table(table, spec=FinalizeSpec(table_key=table_key, mode="tolerant"))
    return result.good


def finalize_analytics_rows(table_key: str, rows: RowInput) -> pa.Table:
    """Build and finalize an analytics table from row payloads.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table built from row inputs.
    """
    table, _ = table_for_rows(table_key, rows)
    return finalize_analytics_table(table_key, table)


__all__ = ["finalize_analytics_rows", "finalize_analytics_table"]
