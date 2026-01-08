"""Typed query-result coercion helpers.

DuckDB returns dynamically typed Python values for scalar queries (`.fetchone()`).
This module provides runtime-checked coercion helpers so call sites do not rely
on unchecked casts.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.columnar.conversion import table_to_frame
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBRelation
from codeintel.core.schemas.row_models import normalize_row_value
from codeintel.core.serialization.msgspec import encode_json_line_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import SupportsFloat, SupportsInt

__all__ = [
    "ScalarCoercionError",
    "coerce_datetime",
    "coerce_float",
    "coerce_int",
    "coerce_literal",
    "coerce_optional_datetime",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_optional_str",
    "coerce_str",
    "iter_json_lines_from_arrow_reader",
    "iter_json_lines_from_relation",
    "iter_records_from_arrow_reader",
    "iter_records_from_relation",
    "iter_tuples_from_arrow_reader",
    "iter_tuples_from_relation",
    "records_from_arrow_batch",
    "records_from_arrow_reader",
    "records_from_arrow_table",
    "records_from_relation",
]

_KIND_FLOAT = "float"
_KIND_INT = "int"
_KIND_LITERAL = "literal"
_KIND_STR = "str"
_KIND_DATETIME = "datetime"


class ScalarCoercionError(TypeError):
    """Raised when a scalar query result cannot be coerced to the expected type."""

    def __init__(self, kind: str, *, ctx: str, value: object) -> None:
        message = f"Failed to coerce {ctx} to {kind}: {value!r} ({type(value).__name__})"
        super().__init__(message)
        self.kind = kind
        self.ctx = ctx
        self.raw_value = value


def coerce_int(value: object, *, ctx: str) -> int:
    """Coerce an arbitrary scalar value to an int with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    int
        Coerced integer value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to an integer.
    """
    if isinstance(value, bool):
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)

    try:
        return int(cast("SupportsInt", value))
    except (TypeError, ValueError) as exc:
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value) from exc


def coerce_float(value: object, *, ctx: str) -> float:
    """Coerce an arbitrary scalar value to a float with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    float
        Coerced float value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to a float.
    """
    if isinstance(value, bool):
        raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value)
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError as exc:
            raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value) from exc

    try:
        return float(cast("SupportsFloat", value))
    except (TypeError, ValueError) as exc:
        raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value) from exc


def coerce_optional_float(value: object | None, *, ctx: str) -> float | None:
    """Coerce a value to float, treating None/NaN as missing.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    float | None
        Coerced float value, or None when the value is missing.
    """
    if value is None:
        return None
    coerced = coerce_float(value, ctx=ctx)
    return None if math.isnan(coerced) else coerced


def coerce_optional_int(value: object | None, *, ctx: str) -> int | None:
    """Coerce a value to int, treating None as missing.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    int | None
        Coerced integer value, or None when the value is missing.
    """
    if value is None:
        return None
    return coerce_int(value, ctx=ctx)


def coerce_str(value: object, *, ctx: str) -> str:
    """Coerce an arbitrary scalar value to a string with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    str
        Coerced string value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to a string.
    """
    if value is None or isinstance(value, bool):
        raise ScalarCoercionError(_KIND_STR, ctx=ctx, value=value)
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScalarCoercionError(_KIND_STR, ctx=ctx, value=value) from exc
    return str(value)


def coerce_optional_str(value: object | None, *, ctx: str) -> str | None:
    """Coerce a value to string, treating None as missing.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    str | None
        Coerced string value, or None when the value is missing.
    """
    if value is None:
        return None
    return coerce_str(value, ctx=ctx)


def coerce_datetime(value: object, *, ctx: str) -> datetime:
    """Coerce a value to datetime with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    datetime
        Coerced datetime value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to a datetime.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        resolved = to_pydatetime()
        if isinstance(resolved, datetime):
            return resolved
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:
            raise ScalarCoercionError(_KIND_DATETIME, ctx=ctx, value=value) from exc
    raise ScalarCoercionError(_KIND_DATETIME, ctx=ctx, value=value)


def coerce_optional_datetime(value: object | None, *, ctx: str) -> datetime | None:
    """Coerce a value to datetime, treating None as missing.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    datetime | None
        Coerced datetime value, or None when the value is missing.
    """
    if value is None:
        return None
    return coerce_datetime(value, ctx=ctx)


def coerce_literal[TStr: str](value: object, *, ctx: str, allowed: Sequence[TStr]) -> TStr:
    """Coerce a value to a specific literal set with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB.
    ctx
        Human-readable context string included in errors.
    allowed
        Sequence of allowed literal values.

    Returns
    -------
    TStr
        Coerced literal value.

    Raises
    ------
    ScalarCoercionError
        If the value is not one of the allowed literals.
    """
    text = coerce_str(value, ctx=ctx)
    if text not in allowed:
        allowed_text = ", ".join(allowed)
        ctx_detail = f"{ctx} (allowed: {allowed_text})"
        raise ScalarCoercionError(_KIND_LITERAL, ctx=ctx_detail, value=text)
    return cast("TStr", text)


def records_from_arrow_batch(
    batch: pa.RecordBatch,
    *,
    columns: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Convert an Arrow record batch to row dictionaries with normalized values.

    Parameters
    ----------
    batch
        Arrow record batch to normalize.
    columns
        Optional column subset/order to apply before conversion.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    if batch.num_rows == 0:
        return []
    table = pa.Table.from_batches([batch])
    frame = table_to_frame(table)
    if columns is not None:
        frame = frame.select(list(columns))
    records = cast("list[dict[str, object]]", frame.to_dicts())
    return _normalize_records(records)


def records_from_arrow_table(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Convert an Arrow table to row dictionaries with normalized values.

    Parameters
    ----------
    table
        Arrow table to normalize.
    columns
        Optional column subset/order to apply before conversion.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    if table.num_rows == 0:
        return []
    frame = table_to_frame(table)
    if columns is not None:
        frame = frame.select(list(columns))
    records = cast("list[dict[str, object]]", frame.to_dicts())
    return _normalize_records(records)


def _raise_if_cancelled(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def iter_records_from_arrow_reader(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield row dictionaries from a RecordBatchReader with normalized values.

    Parameters
    ----------
    reader
        Arrow record batch reader to normalize.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    dict[str, object]
        Normalized row dictionaries with missing values set to None.
    """
    batches: Iterable[pa.RecordBatch] = reader
    for batch in batches:
        _raise_if_cancelled(cancel_check)
        yield from records_from_arrow_batch(batch, columns=columns)


def iter_records_from_relation(
    relation: DuckDBRelation,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield row dictionaries from a DuckDB relation.

    Parameters
    ----------
    relation
        DuckDB relation to stream.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    dict[str, object]
        Normalized row dictionaries with missing values set to None.
    """
    reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    yield from iter_records_from_arrow_reader(
        reader,
        columns=columns,
        cancel_check=cancel_check,
    )


def iter_json_lines_from_arrow_reader(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[str]:
    """Yield JSON Lines strings from a RecordBatchReader.

    Parameters
    ----------
    reader
        Arrow record batch reader to normalize.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    str
        JSON Lines-encoded rows.
    """
    for record in iter_records_from_arrow_reader(
        reader,
        columns=columns,
        cancel_check=cancel_check,
    ):
        yield encode_json_line_text(record)


def iter_json_lines_from_relation(
    relation: DuckDBRelation,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[str]:
    """Yield JSON Lines strings from a DuckDB relation.

    Parameters
    ----------
    relation
        DuckDB relation to stream.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    str
        JSON Lines-encoded rows.
    """
    reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    yield from iter_json_lines_from_arrow_reader(
        reader,
        columns=columns,
        cancel_check=cancel_check,
    )


def iter_tuples_from_arrow_reader(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[tuple[object, ...]]:
    """Yield tuple rows from a RecordBatchReader.

    Parameters
    ----------
    reader
        Arrow record batch reader to normalize.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    tuple[object, ...]
        Row tuples in column order.

    """
    for batch in reader:
        _raise_if_cancelled(cancel_check)
        if batch.num_rows == 0:
            continue
        yield from iter_tuples(batch, columns=columns)


def iter_tuples_from_relation(
    relation: DuckDBRelation,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[tuple[object, ...]]:
    """Yield tuple rows from a DuckDB relation.

    Parameters
    ----------
    relation
        DuckDB relation to stream.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Yields
    ------
    tuple[object, ...]
        Row tuples in column order.
    """
    reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    yield from iter_tuples_from_arrow_reader(
        reader,
        columns=columns,
        cancel_check=cancel_check,
    )


def records_from_arrow_reader(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    """Convert an Arrow RecordBatchReader to row dictionaries with normalized values.

    Parameters
    ----------
    reader
        Arrow record batch reader to normalize.
    columns
        Optional column subset/order to apply before conversion.
    cancel_check
        Optional cancellation hook invoked between batches.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    return list(iter_records_from_arrow_reader(reader, columns=columns, cancel_check=cancel_check))


def records_from_relation(
    relation: DuckDBRelation,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    """Convert a DuckDB relation to row dictionaries with normalized values.

    Parameters
    ----------
    relation
        DuckDB relation to materialize into row dictionaries.
    cancel_check
        Optional cancellation hook invoked between batches.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return records_from_arrow_reader(reader, cancel_check=cancel_check)


def _normalize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for record in records:
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            cleaned[str(key)] = normalize_row_value(_normalize_goid_value(value))
        normalized.append(cleaned)
    return normalized


def _normalize_goid_value(value: object) -> object:
    if isinstance(value, Decimal):
        return int(value)
    return value
