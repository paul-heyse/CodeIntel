"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import polars as pl
import polars.datatypes as pl_datatypes
import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.contracts.types import ContractPolicy
from codeintel.build.tabular.arrow_ops import (
    AlignmentReporter,
    align_tabular_to_contract,
    emit_alignment_report,
)
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_mask
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import TabularInput

Frame = TabularInput
PolarsDataType = pl_datatypes.DataType | pl_datatypes.DataTypeClass


def drop_bad_rows(df: TabularInput, required_cols: tuple[str, ...]) -> TabularInput:
    """Drop rows with nulls in required columns.

    Returns
    -------
    TabularInput
        Filtered LazyFrame/Table or the original input for other data.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        table = _as_arrow_table(df)
        if table is None:
            return df
        if not required_cols:
            return table
        indices = _column_indices(table.schema, required_cols)
        return _filter_table(table, indices)
    if not required_cols:
        return lazyframe
    return lazyframe.drop_nulls(list(required_cols))


def clip_numeric(df: TabularInput, col: str, max_value: float) -> TabularInput:
    """Clip numeric column values to a maximum bound.

    Returns
    -------
    TabularInput
        LazyFrame/Table with the numeric column clipped or the original input.

    Raises
    ------
    ValueError
        If the column is missing or not numeric.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        table = _as_arrow_table(df)
        if table is None:
            return df
        index = _column_indices(table.schema, (col,))[0]
        column_type = table.schema.field(index).type
        if not _is_numeric(column_type):
            msg = f"Unsupported clip column type: {column_type}"
            raise ValueError(msg)
        scalar = _scalar_for_type(max_value, column_type)
        return _clip_table(table, index, scalar)
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
        LazyFrame/Table with null policy applied or the original input.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        table = _as_arrow_table(df)
        if table is None:
            return df
        if policy == "preserve":
            return table
        if policy == "drop_bad_rows":
            if not table.schema.names:
                return table
            indices = list(range(len(table.schema)))
            return _filter_table(table, indices)
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
        LazyFrame/Table with columns reordered or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        table = _as_arrow_table(df)
        if table is None:
            return df
        if not column_order:
            return table
        indices = _column_indices(table.schema, column_order)
        ordered = [table.schema.field(index).name for index in indices]
        return table.select(ordered)
    if not column_order:
        return lazyframe
    return lazyframe.select(_selector_by_name(column_order))


def align_contract_output(
    df: TabularInput,
    *,
    table_key: str,
    target_name: str | None,
    policy: ContractPolicy | None = None,
    reporter: AlignmentReporter | None = emit_alignment_report,
) -> TabularInput:
    """Align Arrow or Polars outputs to the contract schema when possible.

    Returns
    -------
    TabularInput
        Aligned tabular input or the original input for unsupported data.
    """
    return align_tabular_to_contract(
        table_key,
        df,
        target_name=target_name,
        policy=policy,
        reporter=reporter,
    )


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


def _as_arrow_table(df: TabularInput) -> pa.Table | None:
    try:
        return tabular_to_arrow_table(df)
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


def _filter_table(table: pa.Table, indices: Sequence[int]) -> pa.Table:
    mask = _valid_mask(table, indices)
    return table.filter(mask)


def _valid_mask(table: pa.Table, indices: Sequence[int]) -> pa.Array:
    mask = is_valid_mask(table.column(indices[0]))
    for index in indices[1:]:
        mask = and_kleene(mask, is_valid_mask(table.column(index)))
    return mask


def _clip_table(table: pa.Table, index: int, scalar: pa.Scalar) -> pa.Table:
    column = table.column(index)
    condition = pc.call_function("greater", [column, scalar])
    clipped = pc.call_function("if_else", [condition, scalar, column])
    field = table.schema.field(index)
    return table.set_column(index, field.name, clipped)


__all__ = [
    "Frame",
    "align_contract_output",
    "cast_schema",
    "clip_numeric",
    "drop_bad_rows",
    "normalize_nulls",
    "sort_columns",
]
