"""Finalize helpers for analytics tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_reader as _finalize_analytics_reader,
)
from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_result as _finalize_analytics_result,
)
from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_rows as _finalize_analytics_rows,
)
from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_table as _finalize_analytics_table,
)
from codeintel.build.tabular.finalize_ops import FinalizeResult
from codeintel.core.columnar.rows import ColumnarRowBuffer

RowInput = ColumnarRowBuffer | Iterable[Mapping[str, object]] | Iterable[Sequence[object]]


def finalize_analytics_result(table_key: str, table: pa.Table) -> FinalizeResult:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    FinalizeResult
        Finalization result containing good and rejected rows.
    """
    return _finalize_analytics_result(table_key, table)


def finalize_analytics_table(table_key: str, table: pa.Table) -> pa.Table:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table.
    """
    return _finalize_analytics_table(table_key, table)


def finalize_analytics_rows(table_key: str, rows: RowInput) -> pa.Table:
    """Build and finalize an analytics table from row payloads.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table built from row inputs.
    """
    return _finalize_analytics_rows(table_key, rows)


def finalize_analytics_reader(table_key: str, reader: pa.RecordBatchReader) -> pa.Table:
    """Finalize an analytics reader against its contract in tolerant mode.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table.
    """
    return _finalize_analytics_reader(table_key, reader).good


__all__ = [
    "finalize_analytics_reader",
    "finalize_analytics_result",
    "finalize_analytics_rows",
    "finalize_analytics_table",
]
