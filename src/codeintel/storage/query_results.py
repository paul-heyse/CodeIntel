"""Typed query-result coercion helpers.

DuckDB returns dynamically typed Python values for scalar queries (`.fetchone()`).
This module provides runtime-checked coercion helpers so call sites do not rely
on unchecked casts.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import polars as pl
import pyarrow as pa

from codeintel.core.schemas.row_models import normalize_row_value
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBRelation

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsInt

__all__ = [
    "ScalarCoercionError",
    "coerce_float",
    "coerce_int",
    "coerce_optional_float",
    "coerce_optional_int",
    "records_from_arrow_reader",
    "records_from_arrow_table",
    "records_from_relation",
]

_KIND_FLOAT = "float"
_KIND_INT = "int"


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


def records_from_arrow_table(table: pa.Table) -> list[dict[str, object]]:
    """Convert an Arrow table to row dictionaries with normalized values.

    Parameters
    ----------
    table
        Arrow table to normalize.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    if table.num_rows == 0:
        return []
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        frame = frame.to_frame()
    records = cast("list[dict[str, object]]", frame.to_dicts())
    return _normalize_records(records)


def records_from_arrow_reader(reader: pa.RecordBatchReader) -> list[dict[str, object]]:
    """Convert an Arrow RecordBatchReader to row dictionaries with normalized values.

    Parameters
    ----------
    reader
        Arrow record batch reader to normalize.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    records: list[dict[str, object]] = []
    for batch in reader:
        if batch.num_rows == 0:
            continue
        frame = pl.from_arrow(batch)
        if isinstance(frame, pl.Series):
            frame = frame.to_frame()
        batch_records = cast("list[dict[str, object]]", frame.to_dicts())
        records.extend(_normalize_records(batch_records))
    return records


def records_from_relation(relation: DuckDBRelation) -> list[dict[str, object]]:
    """Convert a DuckDB relation to row dictionaries with normalized values.

    Parameters
    ----------
    relation
        DuckDB relation to materialize into row dictionaries.

    Returns
    -------
    list[dict[str, object]]
        List of row dictionaries with missing values set to None.
    """
    reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return records_from_arrow_reader(reader)


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
