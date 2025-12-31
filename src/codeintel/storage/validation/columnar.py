"""Arrow-first validation helpers for columnar data."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.validation.schema_constraints import (
    arrow_batch_errors,
    arrow_table_errors,
    nullability_errors_for_batch,
    nullability_errors_for_table,
    observation_errors_for_batch,
    observation_errors_for_table,
    schema_errors,
)
from codeintel.core.validation.schema_constraints import (
    validate_parquet_path as validate_parquet_constraints,
)
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

ValidationMode = Literal["strict", "warn", "skip"]

LOG = logging.getLogger(__name__)

__all__ = [
    "TableValidationError",
    "ValidationMode",
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
        provider = None
    resolution = resolve_table_schema(table_key, schema_provider=provider)
    return resolution.table_schema


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

    Returns
    -------
    pyarrow.Table
        Validated table, possibly unchanged.
    """
    schema = table_schema or _lookup_table_schema(table_key)
    if schema is None or mode == "skip":
        return table

    errors: list[str] = []
    errors.extend(schema_errors(schema, table.schema))
    errors.extend(arrow_table_errors(table))
    errors.extend(nullability_errors_for_table(schema, table))
    errors.extend(observation_errors_for_table(schema, table, schema_observation))
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

    Returns
    -------
    pyarrow.RecordBatchReader
        Validated reader, possibly unchanged.
    """
    resolved_schema = table_schema or _lookup_table_schema(table_key)
    if resolved_schema is None or mode == "skip":
        return reader

    schema = resolved_schema
    errors = schema_errors(schema, reader.schema)
    _handle_errors(table_key, errors, mode)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch_index, batch in enumerate(reader):
            batch_errors: list[str] = []
            batch_errors.extend(arrow_batch_errors(batch))
            batch_errors.extend(nullability_errors_for_batch(schema, batch))
            batch_errors.extend(observation_errors_for_batch(schema, batch, schema_observation))
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
    """Validate a Parquet file against the registered TableSchema."""
    resolved_schema = table_schema or _lookup_table_schema(table_key)
    if resolved_schema is None or mode == "skip":
        return
    errors = validate_parquet_constraints(
        path,
        table_schema=resolved_schema,
        observation=schema_observation,
    )
    _handle_errors(table_key, errors, mode)
