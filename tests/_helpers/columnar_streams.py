"""Columnar test helpers for building contract-aligned Arrow/Polars streams."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, TypeGuard, cast

import pyarrow as pa

from codeintel.core.columnar.conversion import (
    empty_table_from_schema,
    reader_from_batches,
    table_from_batches,
)
from codeintel.core.columnar.rows import ColumnarRowBuffer, ColumnarRows, columnar_row_count
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.schemas.arrow_gen import DEFAULT_EXTRAS_COLUMN, arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import ColumnType, TableSchema
from codeintel.storage.contracts.schema_provider import get_schema_provider
from tests._helpers.schemas import ensure_storage_contract_catalog

if TYPE_CHECKING:
    import polars as pl

    from codeintel.core.columnar.schema_alignment import ExtrasPolicy
    from codeintel.storage.warehouse import MaterializationResult, MaterializeOptions, Warehouse
else:
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        pl = None

RowsInput = Sequence[Mapping[str, object]] | ColumnarRows


class PolarsUnavailableError(RuntimeError):
    """Raised when Polars is required for columnar test helpers."""


def contract_schema_for_table_key(table_key: str) -> pa.Schema:
    """Return the Arrow contract schema for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    pyarrow.Schema
        Arrow schema for the table contract.
    """
    table_schema = _table_schema_for_key(table_key)
    return arrow_contract_for_table_schema(table_schema=table_schema)


def reader_for_rows(
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
    extras_policy: ExtrasPolicy | None = None,
    validate_contract: bool = True,
) -> pa.RecordBatchReader:
    """Return a contract-aligned RecordBatchReader for row inputs.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row data as mappings or columnar rows.
    columns
        Optional column selection used to align row data.
    extras_policy
        Optional extras policy override for schema alignment.
    validate_contract
        Whether to assert the aligned schema matches the contract schema.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the contract schema.
    """
    table_schema = _table_schema_for_key(table_key)
    buffer = _buffer_for_rows(table_schema, rows, columns=columns)
    contract_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    table = _table_for_buffer(buffer, contract_schema)
    reader = reader_from_batches(table.schema, table.to_batches())
    resolved_policy = extras_policy or extras_policy_from_schema(contract_schema)
    aligned = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=resolved_policy,
    )
    if validate_contract:
        _ensure_contract_alignment(
            table_key=table_schema.table_key,
            reader_schema=aligned.schema,
            contract_schema=contract_schema,
            extras_policy=resolved_policy,
        )
    return aligned


def _table_for_buffer(buffer: ColumnarRowBuffer, contract_schema: pa.Schema) -> pa.Table:
    if buffer.row_count == 0:
        return empty_table_from_schema(contract_schema)
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for name in buffer.columns:
        values = buffer.data[name]
        if name in contract_schema.names:
            field = contract_schema.field(name)
            coerced_values = _coerce_values_for_arrow_type(values, field.type)
            arrays.append(pa.array(coerced_values, type=field.type))
            fields.append(field)
            continue
        array = pa.array(values)
        arrays.append(array)
        fields.append(pa.field(name, array.type))
    batch = pa.record_batch(arrays, schema=pa.schema(fields))
    return table_from_batches([batch], schema=batch.schema)


def _coerce_values_for_arrow_type(
    values: Sequence[object],
    field_type: pa.DataType,
) -> Sequence[object]:
    if pa.types.is_timestamp(field_type):
        return [_coerce_timestamp_value(value) for value in values]
    return values


def _coerce_timestamp_value(value: object) -> object:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def table_for_rows(
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.Table:
    """Return a contract-aligned Arrow table for row inputs.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row data as mappings or columnar rows.
    columns
        Optional column selection used to align row data.
    extras_policy
        Optional extras policy override for schema alignment.

    Returns
    -------
    pyarrow.Table
        Arrow table aligned to the contract schema.
    """
    reader = reader_for_rows(
        table_key,
        rows,
        columns=columns,
        extras_policy=extras_policy,
    )
    return table_from_batches(reader, schema=reader.schema)


def materialize_table_from_rows(
    warehouse: Warehouse,
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
    options: MaterializeOptions | None = None,
) -> MaterializationResult:
    """Materialize row data via the columnar table path.

    Parameters
    ----------
    warehouse
        Warehouse instance handling the materialization.
    table_key
        Fully qualified table key (schema.table).
    rows
        Row data as mappings or columnar rows.
    columns
        Optional column selection used to align row data.
    options
        Optional materialization options for the warehouse.

    Returns
    -------
    MaterializationResult
        Result metadata for the materialized table.
    """
    reader = reader_for_rows(table_key, rows, columns=columns)
    return warehouse.materialize_table(table_key, reader, options=options)


def lazyframe_for_rows(
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> pl.LazyFrame:
    """Return a Polars LazyFrame for row inputs.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row data as mappings or columnar rows.
    columns
        Optional column selection used to align row data.
    extras_policy
        Optional extras policy override for schema alignment.

    Returns
    -------
    polars.LazyFrame
        LazyFrame aligned to the contract schema.

    Raises
    ------
    PolarsUnavailableError
        If Polars is not installed in the runtime environment.
    """
    if pl is None:  # pragma: no cover
        raise PolarsUnavailableError
    table = table_for_rows(table_key, rows, columns=columns, extras_policy=extras_policy)
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


@lru_cache(maxsize=128)
def _table_schema_for_key(table_key: str) -> TableSchema:
    ensure_storage_contract_catalog()
    provider = get_schema_provider()
    table_schema = provider.get_table_schema(table_key)
    if table_schema is None:
        msg = f"Missing table schema for {table_key}"
        raise ValueError(msg)
    return table_schema


def _buffer_for_rows(
    table_schema: TableSchema,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None,
) -> ColumnarRowBuffer:
    buffer = _buffer_for_columns(table_schema, columns=columns)
    json_columns = {column.name for column in table_schema.columns if column.type == "JSON"}
    if _is_columnar_rows(rows):
        return _buffer_from_columnar_rows(buffer, rows, json_columns=json_columns)
    row_sequence = cast("Sequence[object]", rows)
    if not row_sequence:
        return buffer
    if _is_mapping_sequence(row_sequence):
        return _buffer_from_mappings(buffer, row_sequence, json_columns=json_columns)
    message = "Row inputs must be mappings or ColumnarRows."
    raise TypeError(message)


def _buffer_for_columns(
    table_schema: TableSchema,
    *,
    columns: Sequence[str] | None,
) -> ColumnarRowBuffer:
    type_by_name = {column.name: column.type for column in table_schema.columns}
    nullable_by_name = {column.name: column.nullable for column in table_schema.columns}
    resolved_columns = tuple(columns) if columns else tuple(type_by_name)
    default_type: ColumnType = "VARCHAR"
    column_types: tuple[ColumnType, ...] = tuple(
        type_by_name.get(name, default_type) for name in resolved_columns
    )
    return ColumnarRowBuffer(
        table_key=table_schema.table_key,
        columns=resolved_columns,
        column_types=column_types,
        column_nullable=tuple(nullable_by_name.get(name, True) for name in resolved_columns),
        data={name: [] for name in resolved_columns},
    )


def _buffer_from_columnar_rows(
    buffer: ColumnarRowBuffer,
    rows: ColumnarRows,
    *,
    json_columns: set[str],
) -> ColumnarRowBuffer:
    row_count = columnar_row_count(rows)
    for row_idx in range(row_count):
        row_map: dict[str, object] = {}
        for name in buffer.columns:
            values = rows.get(name)
            row_map[name] = values[row_idx] if values is not None else None
        _guard_json_stringification(row_map, json_columns=json_columns)
        buffer.append(row_map)
    return buffer


def _buffer_from_mappings(
    buffer: ColumnarRowBuffer,
    rows: Sequence[Mapping[str, object]],
    *,
    json_columns: set[str],
) -> ColumnarRowBuffer:
    for row in rows:
        row_map = {name: row.get(name) for name in buffer.columns}
        _guard_json_stringification(row_map, json_columns=json_columns)
        buffer.append(row_map)
    return buffer


def _guard_json_stringification(row: Mapping[str, object], *, json_columns: set[str]) -> None:
    if not json_columns:
        return
    for name in json_columns:
        value = row.get(name)
        if not isinstance(value, str):
            continue
        if _looks_like_json(value):
            msg = f"JSON stringification detected for column {name}; pass dict/list instead."
            raise ValueError(msg)


def _looks_like_json(value: str) -> bool:
    raw = value.strip()
    if not raw:
        return False
    if raw[0] not in {"{", "[", '"'}:
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def _ensure_contract_alignment(
    *,
    table_key: str,
    reader_schema: pa.Schema,
    contract_schema: pa.Schema,
    extras_policy: ExtrasPolicy,
) -> None:
    _ensure_schema_metadata(reader_schema, contract_schema)
    _ensure_schema_fields(table_key, reader_schema, contract_schema, extras_policy)


def _ensure_schema_metadata(reader_schema: pa.Schema, contract_schema: pa.Schema) -> None:
    contract_metadata = contract_schema.metadata or {}
    if not contract_metadata:
        return
    reader_metadata = reader_schema.metadata or {}
    for key, value in contract_metadata.items():
        if reader_metadata.get(key) != value:
            msg = f"Contract metadata mismatch for {key!r}"
            raise ValueError(msg)


def _ensure_schema_fields(
    table_key: str,
    reader_schema: pa.Schema,
    contract_schema: pa.Schema,
    extras_policy: ExtrasPolicy,
) -> None:
    expected_names = list(contract_schema.names)
    if extras_policy == "retain":
        extras_column = _extras_column_from_schema(contract_schema)
        if extras_column in reader_schema.names and extras_column not in expected_names:
            expected_names = [*expected_names, extras_column]
    if reader_schema.names != expected_names:
        msg = (
            f"Contract schema mismatch for {table_key}: expected {expected_names}, "
            f"got {list(reader_schema.names)}"
        )
        raise ValueError(msg)
    for field in contract_schema:
        actual = reader_schema.field(field.name)
        if actual.type != field.type:
            msg = (
                f"Contract type mismatch for {table_key}.{field.name}: "
                f"expected {field.type}, got {actual.type}"
            )
            raise ValueError(msg)


def _extras_column_from_schema(schema: pa.Schema, *, default: str = DEFAULT_EXTRAS_COLUMN) -> str:
    metadata = _decode_metadata(schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return default


def _decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def _decode_metadata_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _is_columnar_rows(rows: RowsInput) -> TypeGuard[ColumnarRows]:
    return isinstance(rows, dict)


def _is_mapping_sequence(rows: Sequence[object]) -> TypeGuard[Sequence[Mapping[str, object]]]:
    if not rows:
        return False
    return isinstance(rows[0], Mapping)


__all__ = [
    "contract_schema_for_table_key",
    "lazyframe_for_rows",
    "materialize_table_from_rows",
    "reader_for_rows",
    "table_for_rows",
]
