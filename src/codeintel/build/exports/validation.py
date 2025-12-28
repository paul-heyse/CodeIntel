"""Validate exported JSONL/Parquet datasets against JSON Schemas.

This module provides validation utilities for exported dataset files.
Validation uses generated JSON Schemas from the schema registry.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
from referencing import Registry

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import log_export_error
from codeintel.build.schemas.json_schema_registry import get_json_schema
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.errors.schema import SCHEMA_NOT_FOUND, SCHEMA_VALIDATION_FAILED, SchemaError
from codeintel.core.errors.taxonomy import NOT_FOUND

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def _get_generated_schema(table_key: str) -> dict[str, Any] | None:
    """Get a generated JSON Schema for the table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    dict[str, Any] | None
        Generated JSON Schema, or None if not available.
    """
    try:
        return get_json_schema(table_key)
    except (KeyError, SchemaError) as e:
        log.debug("Schema lookup failed for %s: %s", table_key, e)
        return None


def _normalize_value(value: object) -> object:
    if isinstance(value, Decimal):
        normalized = normalize_decimal_id(value)
        return normalized if normalized is not None else value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(val) for key, val in value.items()}
    return value


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in record.items()}


def _validate_records(
    records: list[dict[str, Any]], validator: jsonschema.Draft202012Validator
) -> list[str]:
    """Validate records against a schema.

    Parameters
    ----------
    records
        List of record dictionaries to validate.
    validator
        JSON Schema validator instance.

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    normalized = [_normalize_record(record) for record in records]
    return [
        f"row={idx}: {error.message}"
        for idx, record in enumerate(normalized)
        for error in validator.iter_errors(record)
    ]


def _validate_jsonl(path: Path, validator: jsonschema.Draft202012Validator) -> list[str]:
    """Validate a JSONL file against a schema.

    Parameters
    ----------
    path
        Path to JSONL file.
    validator
        JSON Schema validator instance.

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"row={idx}: invalid JSON ({exc})")
                continue
            errors.extend(_validate_records([record], validator))
    return errors


def _validate_parquet(path: Path, validator: jsonschema.Draft202012Validator) -> list[str]:
    """Validate a Parquet file against a schema.

    Parameters
    ----------
    path
        Path to Parquet file.
    validator
        JSON Schema validator instance.

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    parquet_file = pq.ParquetFile(path)
    errors: list[str] = []
    for batch in parquet_file.iter_batches():
        records = _records_from_batch(batch)
        errors.extend(_validate_records(records, validator))
    return errors


def _records_from_batch(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    columns = batch.schema.names
    arrays = [batch.column(idx) for idx in range(batch.num_columns)]
    records: list[dict[str, Any]] = []
    for row_idx in range(batch.num_rows):
        records.append({name: arrays[idx][row_idx].as_py() for idx, name in enumerate(columns)})
    return records


def validate_export_files(
    table_key: str,
    paths: list[Path],
    *,
    dataset_name: str | None = None,
) -> int:
    """Validate files against the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    dataset_name
        Optional dataset name used for logging context.
    paths
        List of JSONL or Parquet files to validate.

    Returns
    -------
    int
        0 on success, non-zero on validation failure.

    Raises
    ------
    BuildProblemError
        If no schema is available for the table.
    """
    schema = _get_generated_schema(table_key)
    if schema is None:
        label = dataset_name or table_key
        message = f"No JSON Schema available for table: {table_key}"
        log_export_error(
            SCHEMA_NOT_FOUND,
            message,
            dataset=label,
            table_key=table_key,
        )
        problem = BuildProblemError.from_error_code(
            error_code=SCHEMA_NOT_FOUND,
            detail=message,
            dataset=label,
            table_key=table_key,
        ).problem_detail
        raise BuildProblemError(problem)

    validator = jsonschema.Draft202012Validator(schema, registry=Registry())
    all_errors: list[str] = []
    for path in paths:
        if not path.exists():
            message = f"File not found: {path}"
            log_export_error(
                NOT_FOUND,
                message,
            )
            all_errors.append(message)
            continue
        if path.suffix.lower() == ".jsonl":
            errors = _validate_jsonl(path, validator)
        else:
            errors = _validate_parquet(path, validator)
        all_errors.extend([f"{path}: {err}" for err in errors])

    if all_errors:
        log_export_error(
            SCHEMA_VALIDATION_FAILED,
            "; ".join(all_errors),
            errors=all_errors,
        )
        return 1
    return 0


__all__ = ["validate_export_files"]
