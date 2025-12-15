"""Validate exported JSONL/Parquet datasets against JSON Schemas.

This module provides validation utilities for exported dataset files.
Validation uses generated JSON Schemas from TableSchema definitions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow.parquet as pq
from referencing import Registry

from codeintel.build.exports.common import ExportError, log_export_error

log = logging.getLogger(__name__)


def _get_generated_schema(schema_name: str) -> dict[str, Any] | None:
    """Get a generated JSON Schema for the schema name.

    Parameters
    ----------
    schema_name
        Dataset name (without .json extension).

    Returns
    -------
    dict[str, Any] | None
        Generated JSON Schema, or None if not available.
    """
    try:
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            get_json_schema_for_dataset_name,
        )

        return get_json_schema_for_dataset_name(schema_name)
    except Exception:  # noqa: BLE001
        return None


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
    return [
        f"row={idx}: {error.message}"
        for idx, record in enumerate(records)
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
    table = pq.read_table(path)
    records = table.to_pylist()
    return _validate_records(records, validator)


def validate_export_files(
    schema_name: str,
    paths: list[Path],
) -> int:
    """Validate files against the named schema.

    Parameters
    ----------
    schema_name
        Name of the schema (dataset name).
    paths
        List of JSONL or Parquet files to validate.

    Returns
    -------
    int
        0 on success, non-zero on validation failure.

    Raises
    ------
    ExportError
        If no schema is available for the dataset.
    """
    schema = _get_generated_schema(schema_name)
    if schema is None:
        message = f"No JSON Schema available for dataset: {schema_name}"
        log_export_error(
            code="export.schema_missing",
            title="Schema missing",
            detail=message,
        )
        raise ExportError(message)

    validator = jsonschema.Draft202012Validator(schema, registry=Registry())
    all_errors: list[str] = []
    for path in paths:
        if not path.exists():
            message = f"File not found: {path}"
            log_export_error(
                code="export.file_missing",
                title="Export file missing",
                detail=message,
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
            code="export.validation_failed",
            title="Export validation failed",
            detail="; ".join(all_errors),
            errors=all_errors,
        )
        return 1
    return 0


__all__ = ["validate_export_files"]
