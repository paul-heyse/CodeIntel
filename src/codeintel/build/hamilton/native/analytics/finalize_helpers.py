"""Finalize helpers for analytics tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import pyarrow as pa

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.schemas.primitives import resolve_stable_sort_keys

RowInput = Iterable[Mapping[str, object]] | Iterable[Sequence[object]]
ORDER_ASC: Final = "ascending"


def finalize_analytics_table(table_key: str, table: pa.Table) -> pa.Table:
    """Finalize an analytics table against its contract in tolerant mode.

    Returns
    -------
    pyarrow.Table
        Contract-aligned analytics table.
    """
    order_by = _stable_order_by(table_key)
    result = finalize_table(
        table,
        spec=FinalizeSpec(table_key=table_key, mode="tolerant", order_by=order_by),
    )
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


def _stable_order_by(table_key: str) -> tuple[SortKey, ...]:
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    stable_sort_keys = resolve_stable_sort_keys(table_schema)
    if not stable_sort_keys:
        return ()
    return tuple((key, ORDER_ASC) for key in stable_sort_keys)


__all__ = ["finalize_analytics_rows", "finalize_analytics_table"]
