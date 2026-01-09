"""Deduplication helpers for Arrow and Polars tabular data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import reader_to_table, tabular_to_arrow_reader
from codeintel.build.tabular.kernels import SortKey
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.dedupe_ops import (
    DedupeLegacy,
    DedupeSpec,
)
from codeintel.core.columnar.dedupe_ops import (
    dedupe_table_for_table as _dedupe_table_for_table,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.schemas.service import SchemaService


def _schema_service() -> SchemaService:
    return get_schema_service()


def _schema_service_optional() -> SchemaService | None:
    try:
        return get_schema_service()
    except (RuntimeError, TypeError):
        return None


def _dedupe_sort_columns(
    *,
    available_columns: set[str],
    key_columns: Sequence[str],
    prefer_columns: Sequence[str],
    tie_breakers: Sequence[SortKey],
) -> tuple[list[str], list[bool]]:
    used: set[str] = set()
    columns: list[str] = []
    descending: list[bool] = []
    for name in key_columns:
        if name in available_columns and name not in used:
            used.add(name)
            columns.append(name)
            descending.append(False)
    for name in prefer_columns:
        if name in available_columns and name not in used:
            used.add(name)
            columns.append(name)
            descending.append(True)
    for name, order in tie_breakers:
        if name in available_columns and name not in used:
            used.add(name)
            columns.append(name)
            descending.append(order == "descending")
    return columns, descending


def _merge_prefer_columns(
    spec: DedupeSpec | None,
    prefer_columns: Sequence[str] | None,
) -> DedupeSpec | None:
    if spec is None or not prefer_columns or spec.prefer_columns:
        return spec
    return DedupeSpec(
        keys=spec.keys,
        prefer_columns=tuple(prefer_columns),
        tie_breakers=spec.tie_breakers,
        tier=spec.tier,
        strategy=spec.strategy,
    )


def _resolve_prefer_columns(
    *,
    prefer_columns: Sequence[str] | None,
    spec: DedupeSpec | None,
) -> tuple[Sequence[str], Sequence[SortKey]]:
    if spec is None:
        return tuple(prefer_columns or ()), ()
    prefer = tuple(spec.prefer_columns) if spec.prefer_columns else tuple(prefer_columns or ())
    return prefer, tuple(spec.tie_breakers)


def dedupe_table_for_table(
    table_key: str,
    table: pa.Table,
    *,
    prefer_columns: Sequence[str] | None = None,
    spec: DedupeSpec | None = None,
) -> pa.Table:
    """Return a table with duplicate primary-key rows removed.

    Returns
    -------
    pa.Table
        Table with duplicate primary-key rows removed.
    """
    resolved_spec = _merge_prefer_columns(spec, prefer_columns)
    legacy = None
    if resolved_spec is None and prefer_columns:
        legacy = DedupeLegacy(prefer_columns=tuple(prefer_columns))
    return _dedupe_table_for_table(
        table_key,
        table,
        spec=resolved_spec,
        legacy=legacy,
    )


def _dedupe_lazyframe_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: Sequence[str] | None = None,
    spec: DedupeSpec | None = None,
) -> pl.LazyFrame:
    schema_service = _schema_service_optional()
    schema = schema_service.get_table_schema(table_key) if schema_service is not None else None
    key_columns = list(spec.keys) if spec is not None and spec.keys else []
    if not key_columns and schema is not None and schema.primary_key:
        key_columns = list(schema.primary_key)
    if not key_columns:
        return frame
    try:
        available = set(frame.collect_schema().names())
    except (AttributeError, pl.exceptions.PolarsError, ValueError):
        available = set()
    if not available and schema is not None:
        available = set(schema.column_names())
    if not available:
        available = set(key_columns)
    missing = [name for name in key_columns if name not in available]
    if missing:
        msg = f"Deduplication missing key columns: {missing}"
        raise ValueError(msg)
    resolved_prefer, tie_breakers = _resolve_prefer_columns(
        prefer_columns=prefer_columns,
        spec=spec,
    )
    columns, descending = _dedupe_sort_columns(
        available_columns=available,
        key_columns=key_columns,
        prefer_columns=list(resolved_prefer),
        tie_breakers=tie_breakers,
    )
    if columns:
        frame = frame.sort(by=columns, descending=descending, nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


def dedupe_tabular(
    table_key: str,
    value: InferableTabularInput,
    *,
    prefer_columns: Sequence[str] | None = None,
    spec: DedupeSpec | None = None,
) -> pa.Table | pl.DataFrame | pl.LazyFrame:
    """Return a deduplicated tabular object based on table primary keys.

    Returns
    -------
    deduped : pa.Table | pl.DataFrame | pl.LazyFrame
        Deduplicated table or frame.
    """
    if isinstance(value, pl.LazyFrame):
        return _dedupe_lazyframe_for_table(
            value,
            table_key=table_key,
            prefer_columns=prefer_columns,
            spec=spec,
        )
    if isinstance(value, pl.DataFrame):
        deduped = _dedupe_lazyframe_for_table(
            value.lazy(),
            table_key=table_key,
            prefer_columns=prefer_columns,
            spec=spec,
        )
        return deduped.collect()
    if isinstance(value, pa.Table):
        return dedupe_table_for_table(
            table_key,
            value,
            prefer_columns=prefer_columns,
            spec=spec,
        )
    if isinstance(value, pa.RecordBatchReader):
        table = reader_to_table(value)
        return dedupe_table_for_table(
            table_key,
            table,
            prefer_columns=prefer_columns,
            spec=spec,
        )
    table = reader_to_table(tabular_to_arrow_reader(value))
    return dedupe_table_for_table(
        table_key,
        table,
        prefer_columns=prefer_columns,
        spec=spec,
    )


__all__ = [
    "DedupeLegacy",
    "DedupeSpec",
    "dedupe_table_for_table",
    "dedupe_tabular",
]
