"""Arrow-first validation helpers for columnar data."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.core.schemas.contracts import DEFAULT_EXTRAS_COLUMN
from codeintel.core.schemas.primitives import Column, TableSchema, column_type_base
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

ValidationMode = Literal["strict", "warn", "skip"]

LOG = logging.getLogger(__name__)

__all__ = [
    "TableValidationError",
    "ValidationMode",
    "is_compatible_arrow_type",
    "validate_parquet_path",
    "validate_record_batch_reader",
    "validate_table",
]


class TableValidationError(ValueError):
    """Raised when a table fails columnar validation."""

    def __init__(self, table_key: str, errors: list[str]) -> None:
        message = f"Validation failed for {table_key}: " + "; ".join(errors)
        super().__init__(message)
        self.table_key = table_key
        self.errors = errors


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return None
    return provider.get_table_schema(table_key)


_DECIMAL_PATTERN = re.compile(r"^DECIMAL\\((\\d+),(\\d+)\\)$")


def _decimal_scale_zero(column_type: str) -> bool:
    compact = column_type.upper().replace(" ", "")
    match = _DECIMAL_PATTERN.match(compact)
    if match is None:
        return False
    return int(match.group(2)) == 0


def _is_list_like(data_type: pa.DataType) -> bool:
    checks = [
        pa.types.is_list,
        pa.types.is_large_list,
        pa.types.is_fixed_size_list,
    ]
    list_view = getattr(pa.types, "is_list_view", None)
    if callable(list_view):
        checks.append(list_view)
    large_list_view = getattr(pa.types, "is_large_list_view", None)
    if callable(large_list_view):
        checks.append(large_list_view)
    return any(check(data_type) for check in checks)


def _is_compatible_arrow_type(column: Column, actual_type: pa.DataType) -> bool:
    normalized = _unwrap_dictionary_type(actual_type)
    if pa.types.is_null(normalized):
        return column.nullable
    base = column_type_base(column.type)
    compatibility = _compatibility_for_base(base, column.type, normalized)
    if compatibility is None:
        return True
    return compatibility


def is_compatible_arrow_type(column: Column, actual_type: pa.DataType) -> bool:
    """Return True when the Arrow type is compatible with the column definition.

    Returns
    -------
    bool
        True when the Arrow type is compatible with the column.
    """
    return _is_compatible_arrow_type(column, actual_type)


def _compatibility_for_base(
    base: str,
    column_type: str,
    normalized: pa.DataType,
) -> bool | None:
    checker = _DIRECT_COMPAT_CHECKS.get(base)
    if checker is not None:
        return checker(normalized)
    predicate = _predicate_for_base(base, column_type)
    if predicate is None:
        return None
    return predicate(normalized)


def _predicate_for_base(
    base: str,
    column_type: str,
) -> Callable[[pa.DataType], bool] | None:
    if base == "DECIMAL" and _decimal_scale_zero(column_type):
        return _is_decimal_or_int
    return _BASE_TYPE_PREDICATES.get(base)


def _unwrap_dictionary_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(data_type):
        return data_type.value_type
    return data_type


def _is_decimal_or_int(data_type: pa.DataType) -> bool:
    return pa.types.is_integer(data_type) or pa.types.is_decimal(data_type)


def _is_decimal_or_float(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
        or pa.types.is_integer(data_type)
    )


def _is_string_like(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _is_temporal(data_type: pa.DataType) -> bool:
    return pa.types.is_timestamp(data_type) or pa.types.is_date(data_type)


def _always_true(_: pa.DataType) -> bool:
    return True


def _build_base_type_predicates() -> dict[str, Callable[[pa.DataType], bool]]:
    return {
        "INTEGER": pa.types.is_integer,
        "BIGINT": pa.types.is_integer,
        "DOUBLE": _is_decimal_or_float,
        "DECIMAL": _is_decimal_or_float,
        "BOOLEAN": pa.types.is_boolean,
        "VARCHAR": _is_string_like,
        "TIMESTAMP": _is_temporal,
        "TIMESTAMPTZ": _is_temporal,
    }


_BASE_TYPE_PREDICATES = _build_base_type_predicates()
_DIRECT_COMPAT_CHECKS: dict[str, Callable[[pa.DataType], bool]] = {
    "JSON": _always_true,
    "STRUCT": pa.types.is_struct,
    "LIST": _is_list_like,
    "MAP": pa.types.is_map,
    "UNION": pa.types.is_union,
}


def _is_compatible_type(column: Column, actual_type: pa.DataType) -> bool:
    return _is_compatible_arrow_type(column, actual_type)


def _schema_errors(table_schema: TableSchema, actual_schema: pa.Schema) -> list[str]:
    errors: list[str] = []
    expected_names = [column.name for column in table_schema.columns]
    actual_names = [name for name in actual_schema.names if name != DEFAULT_EXTRAS_COLUMN]

    missing = [name for name in expected_names if name not in actual_names]
    extra = [name for name in actual_names if name not in expected_names]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected columns: {', '.join(extra)}")
    if not missing and not extra and expected_names != actual_names:
        errors.append(
            "Column order mismatch: expected "
            f"{', '.join(expected_names)} but got {', '.join(actual_names)}"
        )

    by_name = {column.name: column for column in table_schema.columns}
    for name in expected_names:
        if name not in actual_schema.names:
            continue
        column = by_name[name]
        actual_field = actual_schema.field(name)
        if not _is_compatible_type(column, actual_field.type):
            errors.append(
                f"Column {name} type mismatch: expected {column.type}, got {actual_field.type}"
            )
    return errors


def _has_nulls(values: pa.Array | pa.ChunkedArray) -> bool:
    null_count = values.null_count
    if null_count is not None and null_count >= 0:
        return null_count > 0
    if isinstance(values, pa.ChunkedArray):
        return any((chunk.null_count or 0) > 0 for chunk in values.chunks)
    return False


def _nullability_errors_for_table(table_schema: TableSchema, table: pa.Table) -> list[str]:
    errors: list[str] = []
    for column in table_schema.columns:
        if column.nullable:
            continue
        if column.name not in table.column_names:
            continue
        if not _all_valid(table.column(column.name)):
            errors.append(f"Column {column.name} contains nulls but is non-nullable")
    return errors


def _nullability_errors_for_batch(
    table_schema: TableSchema,
    batch: pa.RecordBatch,
) -> list[str]:
    errors: list[str] = []
    names = list(batch.schema.names)
    for column in table_schema.columns:
        if column.nullable:
            continue
        if column.name not in names:
            continue
        index = names.index(column.name)
        if not _all_valid(batch.column(index)):
            errors.append(f"Column {column.name} contains nulls but is non-nullable")
    return errors


def _all_valid(values: pa.Array | pa.ChunkedArray) -> bool:
    is_valid = getattr(pc, "is_valid", None)
    if not callable(is_valid):
        return not _has_nulls(values)
    try:
        mask = is_valid(values)
        all_fn = getattr(pc, "all", None)
        if callable(all_fn):
            result = all_fn(mask)
            as_py = getattr(result, "as_py", None)
            if callable(as_py):
                value = as_py()
                return bool(value) if value is not None else False
        return not _has_nulls(values)
    except (TypeError, pa.ArrowInvalid):
        return not _has_nulls(values)


def _arrow_validation_errors(table: pa.Table) -> list[str]:
    try:
        table.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def _arrow_batch_validation_errors(batch: pa.RecordBatch) -> list[str]:
    try:
        batch.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def _handle_errors(table_key: str, errors: list[str], mode: ValidationMode) -> None:
    if not errors or mode == "skip":
        return
    if mode == "warn":
        for error in errors:
            LOG.warning("Validation warning for %s: %s", table_key, error)
        return
    raise TableValidationError(table_key, errors)


def validate_table(
    table_key: str,
    table: pa.Table,
    *,
    table_schema: TableSchema | None = None,
    schema_observation: SchemaObservationRecord | None = None,
    mode: ValidationMode = "strict",
) -> pa.Table:
    """Validate an Arrow table against the registered TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    table
        Arrow table to validate.
    table_schema
        Optional schema override to use instead of the registry lookup.
    schema_observation
        Optional schema observation record for inferred constraint checks.
    mode
        Validation behavior: ``"strict"`` raises, ``"warn"`` logs, ``"skip"`` ignores.

    Returns
    -------
    pa.Table
        The original table when validation succeeds or is skipped.
    """
    schema = table_schema or _lookup_table_schema(table_key)
    if schema is None or mode == "skip":
        return table

    errors = []
    errors.extend(_schema_errors(schema, table.schema))
    errors.extend(_arrow_validation_errors(table))
    errors.extend(_nullability_errors_for_table(schema, table))
    errors.extend(_observation_errors_for_table(schema, table, schema_observation))
    _handle_errors(table_key, errors, mode)
    return table


def validate_record_batch_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    table_schema: TableSchema | None = None,
    schema_observation: SchemaObservationRecord | None = None,
    mode: ValidationMode = "strict",
) -> pa.RecordBatchReader:
    """Validate a RecordBatchReader stream against the registered TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    reader
        RecordBatchReader to validate.
    table_schema
        Optional schema override to use instead of the registry lookup.
    schema_observation
        Optional schema observation record for inferred constraint checks.
    mode
        Validation behavior: ``"strict"`` raises, ``"warn"`` logs, ``"skip"`` ignores.

    Returns
    -------
    pa.RecordBatchReader
        RecordBatchReader that yields validated batches.
    """
    resolved_schema = table_schema or _lookup_table_schema(table_key)
    if resolved_schema is None or mode == "skip":
        return reader

    schema = resolved_schema
    errors = _schema_errors(schema, reader.schema)
    _handle_errors(table_key, errors, mode)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch_index, batch in enumerate(reader):
            batch_errors = []
            batch_errors.extend(_arrow_batch_validation_errors(batch))
            batch_errors.extend(_nullability_errors_for_batch(schema, batch))
            batch_errors.extend(_observation_errors_for_batch(schema, batch, schema_observation))
            if batch_errors:
                batch_errors = [f"batch {batch_index}: {error}" for error in batch_errors]
                _handle_errors(table_key, batch_errors, mode)
            yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


def validate_parquet_path(
    table_key: str,
    path: Path,
    *,
    table_schema: TableSchema | None = None,
    schema_observation: SchemaObservationRecord | None = None,
    mode: ValidationMode = "strict",
) -> None:
    """Validate a Parquet file against the registered TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    path
        Path to the Parquet file.
    table_schema
        Optional schema override to use instead of the registry lookup.
    schema_observation
        Optional schema observation record for inferred constraint checks.
    mode
        Validation behavior: ``"strict"`` raises, ``"warn"`` logs, ``"skip"`` ignores.
    """
    dataset = ds.dataset(str(path), format="parquet")
    reader = dataset.scanner().to_reader()
    validated = validate_record_batch_reader(
        table_key,
        reader,
        table_schema=table_schema,
        schema_observation=schema_observation,
        mode=mode,
    )
    for _batch in validated:
        continue


def _observation_errors_for_table(
    table_schema: TableSchema,
    table: pa.Table,
    observation: SchemaObservationRecord | None,
) -> list[str]:
    stats_by_name = _column_stats_lookup(observation)
    if not stats_by_name:
        return []
    errors: list[str] = []
    for column in table_schema.columns:
        if column.name not in table.column_names:
            continue
        stats = stats_by_name.get(column.name)
        if stats is None:
            continue
        errors.extend(_range_errors(column.name, table.column(column.name), stats))
    return errors


def _observation_errors_for_batch(
    table_schema: TableSchema,
    batch: pa.RecordBatch,
    observation: SchemaObservationRecord | None,
) -> list[str]:
    stats_by_name = _column_stats_lookup(observation)
    if not stats_by_name:
        return []
    errors: list[str] = []
    names = list(batch.schema.names)
    for column in table_schema.columns:
        if column.name not in names:
            continue
        stats = stats_by_name.get(column.name)
        if stats is None:
            continue
        index = names.index(column.name)
        errors.extend(_range_errors(column.name, batch.column(index), stats))
    return errors


def _range_errors(
    name: str,
    values: pa.Array | pa.ChunkedArray,
    stats: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    min_value = stats.get("min")
    max_value = stats.get("max")
    if min_value is not None and _any_out_of_range(values, min_value, op="lt"):
        errors.append(f"Column {name} has values below observed min {min_value}")
    if max_value is not None and _any_out_of_range(values, max_value, op="gt"):
        errors.append(f"Column {name} has values above observed max {max_value}")
    return errors


def _any_out_of_range(
    values: pa.Array | pa.ChunkedArray,
    bound: object,
    *,
    op: Literal["lt", "gt"],
) -> bool:
    scalar = _coerce_scalar(bound, values.type)
    if scalar is None:
        return False
    less = getattr(pc, "less", None)
    greater = getattr(pc, "greater", None)
    compare = less if op == "lt" else greater
    if not callable(compare):
        return False
    try:
        mask = compare(values, scalar)
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False
    return _any_true(mask)


def _any_true(values: pa.Array | pa.ChunkedArray) -> bool:
    any_fn = getattr(pc, "any", None)
    if callable(any_fn):
        result = any_fn(values)
        as_py = getattr(result, "as_py", None)
        if callable(as_py):
            value = as_py()
            return bool(value) if value is not None else False
    return any(values.to_pylist())


def _coerce_scalar(value: object, data_type: pa.DataType) -> pa.Scalar | None:
    try:
        return pa.scalar(value, type=data_type)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


def _column_stats_lookup(
    observation: SchemaObservationRecord | None,
) -> dict[str, Mapping[str, object]]:
    if observation is None:
        return {}
    raw_stats = observation.column_stats
    if not isinstance(raw_stats, Mapping):
        return {}
    stats: dict[str, Mapping[str, object]] = {}
    for name, payload in raw_stats.items():
        if not isinstance(name, str) or not isinstance(payload, Mapping):
            continue
        stats[name] = payload
    return stats
