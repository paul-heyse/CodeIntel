"""Columnar frame helpers for build and analytics pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe
from codeintel.core.columnar.rows import columnar_row_count
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.generated_rows import columns_for_table_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.core.schemas.service import SchemaService

ColumnsSpec = Mapping[str, Sequence[object]] | Sequence[str] | None
JoinStrategy = Literal["inner", "left", "right", "full", "semi", "anti", "cross", "outer"]
JoinValidation = Literal["m:m", "m:1", "1:m", "1:1"]


def _schema_service() -> SchemaService | None:
    try:
        getter = cast(
            "Callable[[], SchemaService]",
            lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None
    try:
        return getter()
    except (RuntimeError, TypeError):
        return None


def _require_schema_service() -> SchemaService:
    service = _schema_service()
    if service is None:
        msg = "SchemaService is required for schema-aligned frames"
        raise RuntimeError(msg)
    return service


def _resolved_columns(*, table_key: str, columns: ColumnsSpec) -> list[str]:
    if isinstance(columns, Mapping):
        if columns:
            return [str(name) for name in columns]
        columns = None
    if columns is not None:
        return [str(name) for name in columns]
    inferred = columns_for_table_key(table_key)
    if inferred is None:
        return []
    return list(inferred)


def _empty_frame_from_columns(columns: Sequence[str]) -> pl.LazyFrame:
    if not columns:
        msg = "Empty frame requires column names for schema-less outputs"
        raise ValueError(msg)
    schema = [(name, pl.Null) for name in columns]
    return pl.DataFrame(schema=schema).lazy()


def _empty_frame_from_schema(
    table_key: str,
    schema_service: SchemaService,
) -> pl.LazyFrame | None:
    arrow_schema = schema_service.get_arrow_schema(table_key)
    if arrow_schema is None:
        try:
            table_schema = schema_service.require_table_schema(table_key)
        except KeyError:
            return None
        arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    reader = pa.RecordBatchReader.from_batches(arrow_schema, [])
    return arrow_reader_to_lazyframe(reader)


def empty_frame_for_table(table_key: str, *, columns: ColumnsSpec = None) -> pl.LazyFrame:
    """Return an empty LazyFrame aligned to the table schema when possible.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    columns
        Optional explicit column order or columnar mapping. Used when schema
        resolution is unavailable.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame with ordered columns.
    """
    schema_service = _schema_service()
    if schema_service is not None:
        frame = _empty_frame_from_schema(table_key, schema_service)
        if frame is not None:
            return frame
    resolved = _resolved_columns(table_key=table_key, columns=columns)
    return _empty_frame_from_columns(resolved)


def empty_lazyframe_for_table(table_key: str) -> pl.LazyFrame:
    """Backward-compatible alias for empty_frame_for_table.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame aligned to the table schema.
    """
    return empty_frame_for_table(table_key)


def rows_to_frame(
    table_key: str,
    rows: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
    *,
    columns: ColumnsSpec = None,
) -> pl.LazyFrame:
    """Convert row sequences into a LazyFrame with schema-ordered columns.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row mappings or row tuples in the expected column order.
    columns
        Optional explicit column order or columnar mapping.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with schema-ordered columns.

    Raises
    ------
    ValueError
        If tuple rows are provided without a column order.
    """
    ordered_columns = _resolved_columns(table_key=table_key, columns=columns)
    if not rows:
        return empty_frame_for_table(table_key, columns=columns)
    first = rows[0]
    if isinstance(first, Mapping):
        frame = pl.DataFrame(rows)
        if not ordered_columns:
            ordered_columns = list(frame.columns)
        missing = [col for col in ordered_columns if col not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(None).alias(col) for col in missing])
        return frame.lazy().select(ordered_columns)
    if not ordered_columns:
        msg = f"Column order required for tuple rows in {table_key}"
        raise ValueError(msg)
    frame = pl.DataFrame(rows, schema=ordered_columns, orient="row")
    return frame.lazy().select(ordered_columns)


def _normalize_columns(columns: ColumnsSpec) -> Mapping[str, Sequence[object]]:
    if columns is None:
        return {}
    if isinstance(columns, Mapping):
        return columns
    return {str(name): [] for name in columns}


def _reader_from_columns(columns: Mapping[str, Sequence[object]]) -> pa.RecordBatchReader:
    payload = {name: list(values) for name, values in columns.items()}
    batch = pa.RecordBatch.from_pydict(payload)
    return pa.RecordBatchReader.from_batches(batch.schema, [batch])


def lazyframe_for_table_columns(
    table_key: str,
    columns: ColumnsSpec,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pl.LazyFrame:
    """Build a LazyFrame aligned to the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    columns
        Columnar mapping of column names to sequences of values, or a list of
        column names for empty frames.
    extras_policy
        Policy for handling extra columns when aligning to the contract schema.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with columns aligned to the schema order.
    """
    normalized = _normalize_columns(columns)
    if not normalized:
        return empty_frame_for_table(table_key)
    row_count = columnar_row_count(normalized)
    if row_count == 0:
        return empty_frame_for_table(table_key)
    schema_service = _require_schema_service()
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
    reader = _reader_from_columns(normalized)
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
    columns: ColumnsSpec,
) -> pl.LazyFrame:
    """Build a LazyFrame for ingest sources, retaining extra fields.

    Returns
    -------
    polars.LazyFrame
        LazyFrame retaining extra ingest columns.
    """
    return lazyframe_for_table_columns(
        table_key,
        columns,
        extras_policy="retain",
    )


@dataclass(frozen=True, slots=True)
class JoinSpec:
    """Join configuration for validated LazyFrame joins."""

    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: JoinStrategy = "inner"
    validate: JoinValidation | None = None
    suffix: str = ""


def join_validated(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec | None = None,
) -> pl.LazyFrame:
    """Join two LazyFrames with optional cardinality validation.

    Returns
    -------
    polars.LazyFrame
        Joined LazyFrame.
    """
    resolved = spec or JoinSpec()
    if resolved.validate is None:
        if resolved.suffix:
            return left.join(
                right,
                on=resolved.on,
                left_on=resolved.left_on,
                right_on=resolved.right_on,
                how=resolved.how,
                suffix=resolved.suffix,
            )
        return left.join(
            right,
            on=resolved.on,
            left_on=resolved.left_on,
            right_on=resolved.right_on,
            how=resolved.how,
        )
    if resolved.suffix:
        return left.join(
            right,
            on=resolved.on,
            left_on=resolved.left_on,
            right_on=resolved.right_on,
            how=resolved.how,
            validate=resolved.validate,
            suffix=resolved.suffix,
        )
    return left.join(
        right,
        on=resolved.on,
        left_on=resolved.left_on,
        right_on=resolved.right_on,
        how=resolved.how,
        validate=resolved.validate,
    )


def dedupe_frame_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Deduplicate rows for a table based on its primary key.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with duplicate primary-key rows removed.
    """
    schema_service = _schema_service()
    schema = schema_service.get_table_schema(table_key) if schema_service is not None else None
    if schema is None or not schema.primary_key:
        return frame
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [column for column in prefer_columns if column in set(schema.column_names())]
        if prefer:
            frame = frame.sort(by=prefer, descending=[True] * len(prefer), nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


def to_records(frame: pl.DataFrame | pl.LazyFrame | pa.Table) -> list[dict[str, Any]]:
    """Convert a columnar frame into a list of dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Row dictionaries converted from the input frame.
    """
    if isinstance(frame, pa.Table):
        resolved = pl.from_arrow(frame)
        if isinstance(resolved, pl.Series):
            resolved = resolved.to_frame()
    elif isinstance(frame, pl.LazyFrame):
        resolved = frame.collect()
    else:
        resolved = frame
    return cast("list[dict[str, Any]]", resolved.to_dicts())


__all__ = [
    "ColumnsSpec",
    "JoinSpec",
    "dedupe_frame_for_table",
    "empty_frame_for_table",
    "empty_lazyframe_for_table",
    "join_validated",
    "lazyframe_for_ingest_columns",
    "lazyframe_for_table_columns",
    "rows_to_frame",
    "to_records",
]
