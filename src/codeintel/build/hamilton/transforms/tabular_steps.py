"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import polars as pl
import polars.datatypes as pl_datatypes
import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular.conversion import tabular_to_arrow_reader
from codeintel.build.tabular.types import TabularInput

Frame = TabularInput
PolarsDataType = pl_datatypes.DataType | pl_datatypes.DataTypeClass


def drop_bad_rows(df: TabularInput, required_cols: tuple[str, ...]) -> TabularInput:
    """Drop rows with nulls in required columns.

    Returns
    -------
    TabularInput
        Filtered LazyFrame/RecordBatchReader or the original input for other data.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        reader = _as_arrow_reader(df)
        if reader is None:
            return df
        if not required_cols:
            return reader
        indices = _column_indices(reader.schema, required_cols)
        return _filter_reader(reader, indices)
    if not required_cols:
        return lazyframe
    return lazyframe.drop_nulls(list(required_cols))


def clip_numeric(df: TabularInput, col: str, max_value: float) -> TabularInput:
    """Clip numeric column values to a maximum bound.

    Returns
    -------
    TabularInput
        LazyFrame/RecordBatchReader with the numeric column clipped or the original input.

    Raises
    ------
    ValueError
        If the column is missing or not numeric.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        reader = _as_arrow_reader(df)
        if reader is None:
            return df
        index = _column_indices(reader.schema, (col,))[0]
        column_type = reader.schema.field(index).type
        if not _is_numeric(column_type):
            msg = f"Unsupported clip column type: {column_type}"
            raise ValueError(msg)
        scalar = _scalar_for_type(max_value, column_type)
        return _clip_reader(reader, index, scalar)
    return lazyframe.with_columns(pl.col(col).clip(upper_bound=max_value))


def _polars_dtype(dtype: str) -> PolarsDataType:
    return pl_datatypes.parse_into_dtype(dtype)


def cast_schema(df: TabularInput, schema: Mapping[str, str]) -> TabularInput:
    """Cast columns to the specified schema mapping when available.

    Returns
    -------
    TabularInput
        LazyFrame with columns cast or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    if not schema:
        return lazyframe
    return lazyframe.with_columns(
        [pl.col(name).cast(_polars_dtype(dtype)) for name, dtype in schema.items()]
    )


def normalize_nulls(df: TabularInput, policy: str) -> TabularInput:
    """Normalize null behavior based on the configured policy.

    Returns
    -------
    TabularInput
        LazyFrame/RecordBatchReader with null policy applied or the original input.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        reader = _as_arrow_reader(df)
        if reader is None:
            return df
        if policy == "preserve":
            return reader
        if policy == "drop_bad_rows":
            if not reader.schema.names:
                return reader
            indices = list(range(len(reader.schema)))
            return _filter_reader(reader, indices)
        msg = f"Unsupported null policy: {policy}"
        raise ValueError(msg)
    if policy == "preserve":
        return lazyframe
    if policy == "drop_bad_rows":
        return lazyframe.drop_nulls()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def sort_columns(df: TabularInput, column_order: Sequence[str]) -> TabularInput:
    """Reorder columns to the provided stable order for Polars data.

    Returns
    -------
    TabularInput
        LazyFrame/RecordBatchReader with columns reordered or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        reader = _as_arrow_reader(df)
        if reader is None:
            return df
        if not column_order:
            return reader
        indices = _column_indices(reader.schema, column_order)
        schema = pa.schema([reader.schema.field(index) for index in indices])
        return _reorder_reader(reader, indices, schema)
    if not column_order:
        return lazyframe
    return lazyframe.select(_selector_by_name(column_order))


def _selector_by_name(names: Sequence[str]) -> pl.Expr | list[str]:
    selectors = getattr(pl, "selectors", None)
    if selectors is None:
        return list(names)
    by_name = getattr(selectors, "by_name", None)
    if callable(by_name):
        return cast("pl.Expr", by_name(list(names)))
    return list(names)


def _as_lazyframe(df: TabularInput) -> pl.LazyFrame | None:
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return None


def _as_arrow_reader(df: TabularInput) -> pa.RecordBatchReader | None:
    try:
        return tabular_to_arrow_reader(df)
    except TypeError:
        return None


def _column_indices(schema: pa.Schema, columns: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for name in columns:
        index = schema.get_field_index(name)
        if index == -1:
            msg = f"Missing column: {name}"
            raise ValueError(msg)
        indices.append(index)
    return indices


def _is_numeric(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)


def _scalar_for_type(value: float, dtype: pa.DataType) -> pa.Scalar:
    try:
        return pa.scalar(value, type=dtype)
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        msg = f"Unsupported clip value {value!r} for {dtype}"
        raise ValueError(msg) from exc


def _filter_reader(
    reader: pa.RecordBatchReader,
    indices: Sequence[int],
) -> pa.RecordBatchReader:
    schema = reader.schema

    def batch_iter() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            mask = _valid_mask(batch, indices)
            yield _filter_batch(batch, mask)

    return pa.RecordBatchReader.from_batches(schema, batch_iter())


def _valid_mask(batch: pa.RecordBatch, indices: Sequence[int]) -> pa.Array:
    first = pc.call_function("is_valid", [batch.column(indices[0])])
    mask = first
    for index in indices[1:]:
        mask = pc.call_function(
            "and_kleene",
            [mask, pc.call_function("is_valid", [batch.column(index)])],
        )
    return mask


def _filter_batch(batch: pa.RecordBatch, mask: pa.Array) -> pa.RecordBatch:
    arrays = [pc.call_function("filter", [column, mask]) for column in batch.columns]
    return pa.RecordBatch.from_arrays(arrays, schema=batch.schema)


def _clip_reader(
    reader: pa.RecordBatchReader,
    index: int,
    scalar: pa.Scalar,
) -> pa.RecordBatchReader:
    schema = reader.schema

    def batch_iter() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            yield _clip_batch(batch, index, scalar)

    return pa.RecordBatchReader.from_batches(schema, batch_iter())


def _clip_batch(
    batch: pa.RecordBatch,
    index: int,
    scalar: pa.Scalar,
) -> pa.RecordBatch:
    arrays = list(batch.columns)
    column = arrays[index]
    condition = pc.call_function("greater", [column, scalar])
    clipped = pc.call_function("if_else", [condition, scalar, column])
    arrays[index] = clipped
    return pa.RecordBatch.from_arrays(arrays, schema=batch.schema)


def _reorder_reader(
    reader: pa.RecordBatchReader,
    indices: Sequence[int],
    schema: pa.Schema,
) -> pa.RecordBatchReader:
    def batch_iter() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            yield _reorder_batch(batch, indices, schema)

    return pa.RecordBatchReader.from_batches(schema, batch_iter())


def _reorder_batch(
    batch: pa.RecordBatch,
    indices: Sequence[int],
    schema: pa.Schema,
) -> pa.RecordBatch:
    arrays = [batch.column(index) for index in indices]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


__all__ = [
    "Frame",
    "cast_schema",
    "clip_numeric",
    "drop_bad_rows",
    "normalize_nulls",
    "sort_columns",
]
