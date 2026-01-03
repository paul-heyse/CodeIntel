"""Validate exported JSONL/Parquet datasets against TableSchema contracts.

This module validates exported dataset files using the authoritative TableSchema,
with Arrow structural checks and optional Pandera semantic validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import log_export_error
from codeintel.build.schemas import get_schema_provider
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.errors.schema import SCHEMA_NOT_FOUND, SCHEMA_VALIDATION_FAILED
from codeintel.core.errors.taxonomy import NOT_FOUND
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import TableSchema, column_type_base
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.core.validation.pandera_schema import (
    PanderaDiagnostics,
    pandera_available,
    pandera_error_diagnostics,
    pandera_error_types,
    pandera_schema_for_table,
)
from codeintel.core.validation.profiles import (
    ValidationProfile,
    normalize_validation_profile,
    resolve_validation_depth,
)
from codeintel.core.validation.schema_constraints import (
    arrow_batch_errors,
    nullability_errors_for_batch,
    observation_errors_for_batch,
    schema_errors,
    schema_metadata_errors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.gateway import BuildGateway

log = logging.getLogger(__name__)


def _table_schema_for_key(
    table_key: str,
    *,
    dataset_name: str | None = None,
) -> TableSchema:
    provider = get_schema_provider()
    schema = provider.get_table_schema(table_key)
    if schema is not None:
        return schema
    label = dataset_name or table_key
    message = f"No TableSchema available for table: {table_key}"
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


def _read_jsonl_records(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    records: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"row={idx}: invalid JSON ({exc})")
                continue
            if not isinstance(payload, dict):
                errors.append(f"row={idx}: expected JSON object")
                continue
            records.append((idx, payload))
    return records, errors


def _normalize_record(
    record: dict[str, Any],
    column_types: dict[str, str],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for name, column_type in column_types.items():
        value = record.get(name)
        if _is_temporal_type(column_type):
            normalized[name] = _coerce_timestamp(value)
        else:
            normalized[name] = normalize_row_value_for_type(value, column_type)
    for key, value in record.items():
        key_str = str(key)
        if key_str in normalized:
            continue
        normalized[key_str] = value
    return normalized


def _is_temporal_type(column_type: str) -> bool:
    base = column_type_base(column_type)
    return base in {"TIMESTAMP", "TIMESTAMPTZ"}


def _coerce_timestamp(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        normalized = trimmed.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return value
    return value


def _reader_for_jsonl(
    path: Path,
    *,
    table_schema: TableSchema,
    contract_schema: pa.Schema,
) -> tuple[pa.RecordBatchReader, list[str]]:
    raw_records, errors = _read_jsonl_records(path)
    column_types = {column.name: column.type for column in table_schema.columns}
    normalized_records: list[dict[str, object]] = []
    for idx, record in raw_records:
        try:
            normalized_records.append(_normalize_record(record, column_types))
        except (TypeError, ValueError) as exc:
            errors.append(f"row={idx}: normalization failed ({exc})")
    if not normalized_records:
        reader = pa.RecordBatchReader.from_batches(contract_schema, [])
        return reader, errors
    try:
        table = pa.Table.from_pylist(normalized_records)
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        errors.append(f"Failed to build Arrow table from JSONL: {exc}")
        reader = pa.RecordBatchReader.from_batches(contract_schema, [])
        return reader, errors
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    return reader, errors


def _reader_for_parquet(path: Path) -> pa.RecordBatchReader:
    parquet_file = pq.ParquetFile(path)
    return pa.RecordBatchReader.from_batches(
        parquet_file.schema_arrow,
        parquet_file.iter_batches(batch_size=DEFAULT_ARROW_BATCH_SIZE),
    )


def _pandera_error_messages(diagnostics: PanderaDiagnostics) -> list[str]:
    payload = diagnostics.to_dict()
    message = f"Pandera validation failed: {payload.get('error')}"
    failure_cases = payload.get("failure_cases")
    if failure_cases is not None:
        message = f"{message}; failure_cases={failure_cases}"
    return [message]


def _pandera_batch_errors(
    *,
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
        diagnostics = pandera_error_diagnostics(exc, table_key=table_key)
        return _pandera_error_messages(diagnostics)
    return []


def _validation_depth(validation_profile: ValidationProfile | None) -> str:
    if validation_profile is None:
        return "data-strict"
    normalized = normalize_validation_profile(validation_profile, default="strict")
    return resolve_validation_depth(normalized)


def _pandera_schema(
    *,
    table_schema: TableSchema,
    contract_schema: pa.Schema,
    validation_profile: ValidationProfile | None,
) -> object | None:
    if not pandera_available():
        return None
    extras_policy = extras_policy_from_schema(contract_schema)
    return pandera_schema_for_table(
        table_schema=table_schema,
        observation=None,
        extras_policy=extras_policy,
        validation_profile=validation_profile,
    )


@dataclass(frozen=True, slots=True)
class _ReaderValidationContext:
    table_key: str
    reader: pa.RecordBatchReader
    table_schema: TableSchema
    contract_schema: pa.Schema
    include_data_checks: bool
    pandera_schema: object | None


def _validate_reader(context: _ReaderValidationContext) -> list[str]:
    errors: list[str] = []
    schema_for_errors = context.reader.schema
    if context.contract_schema.metadata is not None:
        schema_for_errors = context.reader.schema.with_metadata(context.contract_schema.metadata)
    errors.extend(schema_errors(context.table_schema, schema_for_errors))
    errors.extend(schema_metadata_errors(context.reader.schema))
    try:
        aligned = align_reader_to_contract(context.reader, context.contract_schema)
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        errors.append(f"Failed to align to contract schema: {exc}")
        return errors
    try:
        for batch in aligned:
            errors.extend(arrow_batch_errors(batch))
            if not context.include_data_checks:
                continue
            if context.pandera_schema is not None:
                errors.extend(
                    _pandera_batch_errors(
                        table_key=context.table_key,
                        batch=batch,
                        schema=context.pandera_schema,
                    )
                )
            else:
                errors.extend(nullability_errors_for_batch(context.table_schema, batch))
                errors.extend(observation_errors_for_batch(context.table_schema, batch, None))
    except (TypeError, ValueError, pa.ArrowInvalid) as exc:
        errors.append(f"Arrow validation failed: {exc}")
    return errors


def validate_export_files(
    table_key: str,
    paths: list[Path],
    *,
    dataset_name: str | None = None,
    gateway: BuildGateway | None = None,
    validation_profile: ValidationProfile | None = None,
) -> int:
    """Validate files against the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    dataset_name
        Optional dataset name used for logging context.
    gateway
        Optional storage gateway (unused; retained for compatibility).
    paths
        List of JSONL or Parquet files to validate.
    validation_profile
        Optional validation profile that controls schema vs data checks.

    Returns
    -------
    int
        0 on success, non-zero on validation failure.

    """
    _ = gateway
    table_schema = _table_schema_for_key(table_key, dataset_name=dataset_name)
    contract_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    depth = _validation_depth(validation_profile)
    include_data_checks = depth != "schema-only"
    pandera_schema = (
        _pandera_schema(
            table_schema=table_schema,
            contract_schema=contract_schema,
            validation_profile=validation_profile,
        )
        if include_data_checks
        else None
    )

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
        try:
            if path.suffix.lower() == ".jsonl":
                reader, read_errors = _reader_for_jsonl(
                    path,
                    table_schema=table_schema,
                    contract_schema=contract_schema,
                )
                errors = [*read_errors]
                errors.extend(
                    _validate_reader(
                        _ReaderValidationContext(
                            table_key=table_key,
                            reader=reader,
                            table_schema=table_schema,
                            contract_schema=contract_schema,
                            include_data_checks=include_data_checks,
                            pandera_schema=pandera_schema,
                        )
                    )
                )
            else:
                reader = _reader_for_parquet(path)
                errors = _validate_reader(
                    _ReaderValidationContext(
                        table_key=table_key,
                        reader=reader,
                        table_schema=table_schema,
                        contract_schema=contract_schema,
                        include_data_checks=include_data_checks,
                        pandera_schema=pandera_schema,
                    )
                )
        except (OSError, ValueError, TypeError, pa.ArrowInvalid) as exc:
            errors = [f"Failed to read {path.name}: {exc}"]
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
