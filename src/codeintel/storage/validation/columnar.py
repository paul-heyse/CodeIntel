"""Arrow-first validation helpers for columnar data."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

from codeintel.core.columnar.schema_alignment import extras_policy_from_schema
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.validation.pandera_schema import (
    PanderaDiagnostics,
    pandera_available,
    pandera_error_diagnostics,
    pandera_error_types,
    pandera_schema_for_table,
    resolve_extras_policy,
)
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
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord

ValidationMode = Literal["strict", "warn", "skip"]

LOG = logging.getLogger(__name__)

__all__ = [
    "ColumnarValidationContext",
    "TableValidationError",
    "ValidationMode",
    "validate_parquet_path",
    "validate_record_batch_reader",
    "validate_table",
]


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
    context: ColumnarValidationContext | None = None,
    mode: ValidationMode = "strict",
) -> pa.Table:
    """Validate an Arrow table against the registered TableSchema.

    Returns
    -------
    pyarrow.Table
        Validated table, possibly unchanged.
    """
    resolved_context = context or ColumnarValidationContext()
    schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    schema_observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, depth = _resolve_validation_settings(validation_profile, mode)
    if schema is None or resolved_mode == "skip":
        return table

    errors: list[str] = []
    errors.extend(schema_errors(schema, table.schema))
    errors.extend(schema_metadata_errors(table.schema))
    errors.extend(arrow_table_errors(table))
    if depth != "schema-only":
        pandera_schema = _pandera_schema(
            schema,
            table.schema,
            schema_observation,
            validation_profile=validation_profile,
        )
        if pandera_schema is not None:
            errors.extend(_pandera_table_errors(table_key, table, pandera_schema))
        else:
            errors.extend(nullability_errors_for_table(schema, table))
            errors.extend(observation_errors_for_table(schema, table, schema_observation))
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
        Validated reader, possibly unchanged.
    """
    resolved_context = context or ColumnarValidationContext()
    resolved_schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    schema_observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, depth = _resolve_validation_settings(validation_profile, mode)
    if resolved_schema is None or resolved_mode == "skip":
        return reader

    schema = resolved_schema
    errors = schema_errors(schema, reader.schema)
    errors.extend(schema_metadata_errors(reader.schema))
    _handle_errors(table_key, errors, resolved_mode)
    if depth == "schema-only":
        return reader
    pandera_schema = _pandera_schema(
        schema,
        reader.schema,
        schema_observation,
        validation_profile=validation_profile,
    )
    if pandera_schema is not None and schema.primary_key and depth == "data-strict":
        table = pa.Table.from_batches(list(reader))
        table_errors: list[str] = []
        table_errors.extend(arrow_table_errors(table))
        table_errors.extend(_pandera_table_errors(table_key, table, pandera_schema))
        _handle_errors(table_key, table_errors, resolved_mode)
        return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch_index, batch in enumerate(reader):
            batch_errors: list[str] = []
            batch_errors.extend(arrow_batch_errors(batch))
            if pandera_schema is not None:
                batch_errors.extend(_pandera_batch_errors(table_key, batch, pandera_schema))
            else:
                batch_errors.extend(nullability_errors_for_batch(schema, batch))
                batch_errors.extend(observation_errors_for_batch(schema, batch, schema_observation))
            if batch_errors:
                batch_errors = [f"batch {batch_index}: {error}" for error in batch_errors]
                _handle_errors(table_key, batch_errors, resolved_mode)
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
    resolved_schema = resolved_context.table_schema or _lookup_table_schema(table_key)
    schema_observation = resolved_context.schema_observation
    validation_profile = resolved_context.validation_profile
    resolved_mode, _ = _resolve_validation_settings(validation_profile, mode)
    if resolved_schema is None or resolved_mode == "skip":
        return
    errors = validate_parquet_constraints(
        path,
        table_schema=resolved_schema,
        observation=schema_observation,
        validation_profile=validation_profile,
    )
    _handle_errors(table_key, errors, resolved_mode)


def _pandera_schema(
    table_schema: TableSchema,
    arrow_schema: pa.Schema,
    observation: SchemaObservationRecord | None,
    validation_profile: ValidationProfile | None,
) -> object | None:
    if not pandera_available():
        return None
    extras_policy = resolve_extras_policy(
        observation,
        fallback=extras_policy_from_schema(arrow_schema),
    )
    return pandera_schema_for_table(
        table_schema=table_schema,
        observation=observation,
        extras_policy=extras_policy,
        validation_profile=validation_profile,
    )


def _pandera_table_errors(
    table_key: str,
    table: pa.Table,
    schema: object,
) -> list[str]:
    if pl is None:
        return []
    error_types = pandera_error_types()
    if not error_types:
        return []
    frame = pl.from_arrow(table)
    if not isinstance(frame, pl.DataFrame):
        return []
    validate = getattr(schema, "validate", None)
    if not callable(validate):
        return []
    try:
        validate(frame, lazy=True)
    except error_types as exc:
        return _pandera_error_messages(pandera_error_diagnostics(exc, table_key=table_key))
    return []


def _pandera_batch_errors(
    table_key: str,
    batch: pa.RecordBatch,
    schema: object,
) -> list[str]:
    if pl is None:
        return []
    error_types = pandera_error_types()
    if not error_types:
        return []
    frame = pl.from_arrow(pa.Table.from_batches([batch]))
    if not isinstance(frame, pl.DataFrame):
        return []
    validate = getattr(schema, "validate", None)
    if not callable(validate):
        return []
    try:
        validate(frame, lazy=True)
    except error_types as exc:
        return _pandera_error_messages(pandera_error_diagnostics(exc, table_key=table_key))
    return []


def _pandera_error_messages(diagnostics: PanderaDiagnostics) -> list[str]:
    payload = diagnostics.to_dict()
    message = f"Pandera validation failed: {payload.get('error')}"
    failure_cases = payload.get("failure_cases")
    if failure_cases is not None:
        message = f"{message}; failure_cases={failure_cases}"
    return [message]


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
