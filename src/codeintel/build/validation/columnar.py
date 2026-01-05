"""Build-time columnar validation without Pandera dependencies."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import msgspec
import pyarrow as pa

from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.profiles import (
    ValidationDepth,
    ValidationProfile,
    is_lenient_profile,
    normalize_validation_profile,
    resolve_validation_depth,
)
from codeintel.core.validation.schema_constraints import (
    arrow_batch_errors,
    arrow_table_errors,
    nullability_errors_for_batch,
    nullability_errors_for_table,
    observation_errors_for_batch,
    observation_errors_for_table,
    schema_errors,
    schema_metadata_errors,
)
from codeintel.core.validation.schema_constraints import (
    validate_parquet_path as validate_parquet_constraints,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord

ValidationMode = Literal["strict", "warn", "skip"]

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ColumnarValidationContext:
    """Context for columnar validation helpers."""

    table_schema: TableSchema | None = None
    schema_observation: SchemaObservationRecord | None = None
    validation_profile: ValidationProfile | None = None


class TableValidationError(ValueError):
    """Raised when a table fails columnar validation."""

    def __init__(self, table_key: str, errors: list[str]) -> None:
        message = f"Validation failed for {table_key}: " + "; ".join(errors)
        super().__init__(message)
        self.table_key = table_key
        self.errors = errors


@dataclass(slots=True)
class _UniqueKeyState:
    key_columns: tuple[str, ...]
    seen: set[object] = field(default_factory=set)

    def check_batch(self, batch: pa.RecordBatch) -> list[str]:
        indices = _column_indices(batch.schema, self.key_columns)
        if indices is None:
            return []
        arrays = [batch.column(index) for index in indices]
        fields = [batch.schema.field(index) for index in indices]
        struct_array = pa.StructArray.from_arrays(arrays, fields=fields)
        duplicates = 0
        nulls = 0
        for scalar in struct_array:
            if not scalar.is_valid:
                nulls += 1
                continue
            payload = scalar.as_py()
            if payload is None:
                nulls += 1
                continue
            if not isinstance(payload, dict):
                nulls += 1
                continue
            key = tuple(payload.get(name) for name in self.key_columns)
            if any(value is None for value in key):
                nulls += 1
                continue
            marker = _hashable_key(key)
            if marker in self.seen:
                duplicates += 1
            else:
                self.seen.add(marker)
        errors: list[str] = []
        if nulls:
            errors.append(f"Primary key contains {nulls} null values")
        if duplicates:
            errors.append(f"Primary key contains {duplicates} duplicate values")
        return errors


def validate_table(
    table_key: str,
    table: pa.Table,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> pa.Table:
    """Validate an Arrow table against the registered TableSchema.

    Returns
    -------
    pyarrow.Table
        The input table when validation passes or is skipped.
    """
    resolved_context = context or ColumnarValidationContext()
    schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, depth = _resolve_validation_settings(validation_profile, mode)
    if schema is None or resolved_mode == "skip":
        return table

    errors: list[str] = []
    errors.extend(schema_errors(schema, table.schema))
    errors.extend(schema_metadata_errors(table.schema))
    errors.extend(arrow_table_errors(table))
    if depth != "schema-only":
        errors.extend(nullability_errors_for_table(schema, table))
        errors.extend(observation_errors_for_table(schema, table, observation))
        if depth == "data-strict" and schema.primary_key:
            unique_state = _UniqueKeyState(tuple(schema.primary_key))
            for batch in table.to_batches():
                errors.extend(unique_state.check_batch(batch))
                if errors:
                    break
    _handle_errors(table_key, errors, resolved_mode)
    return table


def validate_record_batch_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> pa.RecordBatchReader:
    """Validate a RecordBatchReader stream against the registered TableSchema.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader that yields validated record batches.
    """
    resolved_context = context or ColumnarValidationContext()
    schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, depth = _resolve_validation_settings(validation_profile, mode)
    if schema is None or resolved_mode == "skip":
        return reader

    table_schema: TableSchema = schema
    errors = schema_errors(table_schema, reader.schema)
    errors.extend(schema_metadata_errors(reader.schema))
    _handle_errors(table_key, errors, resolved_mode)
    if depth == "schema-only":
        return reader

    unique_state = None
    if depth == "data-strict" and table_schema.primary_key:
        unique_state = _UniqueKeyState(tuple(table_schema.primary_key))

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch_index, batch in enumerate(reader):
            batch_errors: list[str] = []
            batch_errors.extend(arrow_batch_errors(batch))
            batch_errors.extend(nullability_errors_for_batch(table_schema, batch))
            batch_errors.extend(observation_errors_for_batch(table_schema, batch, observation))
            if unique_state is not None:
                batch_errors.extend(unique_state.check_batch(batch))
            if batch_errors:
                prefixed = [f"batch {batch_index}: {error}" for error in batch_errors]
                _handle_errors(table_key, prefixed, resolved_mode)
            yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


def validate_parquet_path(
    table_key: str,
    path: Path,
    *,
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> None:
    """Validate a Parquet file against the registered TableSchema."""
    resolved_context = context or ColumnarValidationContext()
    schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, _ = _resolve_validation_settings(validation_profile, mode)
    if schema is None or resolved_mode == "skip":
        return
    errors = validate_parquet_constraints(
        path,
        table_schema=schema,
        observation=observation,
        validation_profile=validation_profile,
    )
    _handle_errors(table_key, errors, resolved_mode)


def _lookup_table_schema(table_key: str) -> TableSchema | None:
    try:
        service = get_schema_service()
    except RuntimeError:
        return None
    return service.get_table_schema(table_key)


def _handle_errors(table_key: str, errors: list[str], mode: ValidationMode) -> None:
    if not errors or mode == "skip":
        return
    if mode == "warn":
        for error in errors:
            LOG.warning("Validation warning for %s: %s", table_key, error)
        return
    raise TableValidationError(table_key, errors)


def _resolve_validation_settings(
    validation_profile: ValidationProfile | None,
    mode: ValidationMode,
) -> tuple[ValidationMode, ValidationDepth]:
    if validation_profile is None:
        return mode, "data-strict"
    normalized = normalize_validation_profile(validation_profile, default="strict")
    depth = resolve_validation_depth(normalized)
    if mode == "skip":
        return "skip", depth
    if is_lenient_profile(normalized):
        return "warn", depth
    return "strict", depth


def _hashable_key(values: tuple[object, ...]) -> object:
    try:
        hash(values)
    except TypeError:
        try:
            return msgspec.msgpack.encode(values)
        except TypeError:
            return tuple(repr(item) for item in values)
    else:
        return values


def _column_indices(schema: pa.Schema, names: Sequence[str]) -> tuple[int, ...] | None:
    indices: list[int] = []
    for name in names:
        try:
            index = schema.get_field_index(name)
        except (KeyError, ValueError):
            return None
        if index < 0:
            return None
        indices.append(index)
    return tuple(indices)


__all__ = [
    "ColumnarValidationContext",
    "TableValidationError",
    "ValidationMode",
    "validate_parquet_path",
    "validate_record_batch_reader",
    "validate_table",
]
