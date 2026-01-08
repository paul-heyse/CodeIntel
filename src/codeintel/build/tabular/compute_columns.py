"""Column helpers for constant-valued Arrow table construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from codeintel.build.tabular.compute_helpers import array_from_compute


def empty_table(columns: Sequence[str]) -> pa.Table:
    """Return an empty table with null-typed columns.

    Returns
    -------
    pa.Table
        Empty Arrow table with the requested columns.
    """
    arrays = [pa.array([], type=pa.null()) for _ in columns]
    return pa.Table.from_arrays(arrays, names=list(columns))


def _empty_constant_array(value: object) -> pa.Array | pa.ChunkedArray:
    if value is None:
        return pa.nulls(0)
    try:
        return pa.array([], type=pa.scalar(value).type)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ):
        return pa.array([], type=pa.null())


def _constant_array_via_compute(value: object, length: int) -> pa.Array | pa.ChunkedArray | None:
    return array_from_compute(
        "if_else",
        [pa.scalar(value=True), pa.scalar(value), pa.scalar(value)],
        length=length,
    )


def constant_array(value: object, length: int) -> pa.Array | pa.ChunkedArray:
    """Build a constant-valued array with Arrow compute fallbacks.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Array of the requested length filled with the constant value.
    """
    if length == 0:
        return _empty_constant_array(value)
    if value is None:
        return pa.nulls(length)
    result = _constant_array_via_compute(value, length)
    if result is not None:
        return result
    return pa.array([value] * length)


def append_constant_columns(table: pa.Table, constants: Mapping[str, object]) -> pa.Table:
    """Append constant-valued columns when they are missing.

    Returns
    -------
    pa.Table
        Table with constant-valued columns appended when missing.
    """
    if not constants:
        return table
    existing = set(table.column_names)
    for name, value in constants.items():
        if name in existing:
            continue
        table = table.append_column(name, constant_array(value, table.num_rows))
        existing.add(name)
    return table


__all__ = ["append_constant_columns", "constant_array", "empty_table"]
