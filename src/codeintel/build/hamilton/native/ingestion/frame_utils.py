"""Helpers for building columnar ingestion frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe
from codeintel.core.columnar.rows import columnar_row_count
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)


def empty_lazyframe_for_table(table_key: str) -> pl.LazyFrame:
    """Return an empty LazyFrame using the table schema.

    Returns
    -------
    pl.LazyFrame
        Empty LazyFrame with the table's schema applied.
    """
    schema_service = get_schema_service()
    arrow_schema = schema_service.get_arrow_schema(table_key)
    if arrow_schema is None:
        schema = schema_service.require_table_schema(table_key)
        arrow_schema = arrow_contract_for_table_schema(table_schema=schema)
    reader = pa.RecordBatchReader.from_batches(arrow_schema, [])
    return arrow_reader_to_lazyframe(reader)


def lazyframe_for_table_columns(
    table_key: str,
    columns: Mapping[str, Sequence[object]],
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pl.LazyFrame:
    """Build a LazyFrame for columnar data using the schema's column order.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    columns
        Columnar mapping of column names to sequences of values.
    extras_policy
        Policy for handling extra columns when aligning to the contract schema.
        When None, resolve from Arrow schema metadata.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with columns aligned to the schema order.

    """
    if not columns:
        return empty_lazyframe_for_table(table_key)
    row_count = columnar_row_count(columns)
    if row_count == 0:
        return empty_lazyframe_for_table(table_key)
    schema_service = get_schema_service()
    contract_schema = schema_service.get_arrow_schema(table_key)
    if contract_schema is None:
        table_schema = schema_service.require_table_schema(table_key)
        metadata = (
            ArrowSchemaMetadata(extras_policy=extras_policy) if extras_policy is not None else None
        )
        contract_schema = arrow_contract_for_table_schema(
            table_schema=table_schema,
            metadata=metadata,
        )
    reader = _reader_from_columns(columns)
    resolved_policy = (
        extras_policy if extras_policy is not None else extras_policy_from_schema(contract_schema)
    )
    aligned = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=resolved_policy,
    )
    return arrow_reader_to_lazyframe(aligned)


def lazyframe_for_ingest_columns(
    table_key: str,
    columns: Mapping[str, Sequence[object]],
) -> pl.LazyFrame:
    """Build a LazyFrame for ingest sources, retaining extra fields.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    columns
        Columnar mapping of column names to sequences of values.

    Returns
    -------
    pl.LazyFrame
        LazyFrame aligned to the contract schema with extras retained.
    """
    return lazyframe_for_table_columns(
        table_key,
        columns,
        extras_policy="retain",
    )


def _reader_from_columns(columns: Mapping[str, Sequence[object]]) -> pa.RecordBatchReader:
    payload = {name: list(values) for name, values in columns.items()}
    batch = pa.RecordBatch.from_pydict(payload)
    return pa.RecordBatchReader.from_batches(batch.schema, [batch])


def dedupe_frame_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Deduplicate rows for a table based on its primary key.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with primary-key duplicates removed.
    """
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return frame
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [column for column in prefer_columns if column in set(schema.column_names())]
        if prefer:
            frame = frame.sort(by=prefer, descending=[True] * len(prefer), nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


__all__ = [
    "dedupe_frame_for_table",
    "empty_lazyframe_for_table",
    "lazyframe_for_ingest_columns",
    "lazyframe_for_table_columns",
]
