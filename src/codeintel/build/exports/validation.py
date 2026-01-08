"""Validate exported JSONL/Parquet datasets against TableSchema contracts.

This module validates exported dataset files using the authoritative TableSchema
with Arrow structural checks and observation-based constraints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.json as pa_json

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import log_export_error
from codeintel.build.schemas import get_schema_provider
from codeintel.build.validation.columnar import (
    ColumnarValidationContext,
    TableValidationError,
    validate_parquet_path,
    validate_record_batch_reader,
)
from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.schema_alignment import extras_policy_from_schema
from codeintel.core.errors.schema import SCHEMA_NOT_FOUND, SCHEMA_VALIDATION_FAILED
from codeintel.core.errors.taxonomy import NOT_FOUND
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.validation.profiles import ValidationProfile

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


def _reader_for_jsonl(
    path: Path,
    *,
    contract_schema: pa.Schema,
) -> tuple[pa.RecordBatchReader, list[str]]:
    behavior = _unexpected_field_behavior(contract_schema)
    parse_options = pa_json.ParseOptions(
        explicit_schema=contract_schema,
        unexpected_field_behavior=behavior,
    )
    try:
        return pa_json.open_json(path, parse_options=parse_options), []
    except (pa.ArrowInvalid, OSError, ValueError) as exc:
        errors = [f"Failed to parse JSONL: {exc}"]
        empty_table = pa.Table.from_batches([], schema=contract_schema)
        reader = table_to_reader(empty_table, batch_size=None)
        return reader, errors


def _unexpected_field_behavior(schema: pa.Schema) -> str:
    if extras_policy_from_schema(schema) == "reject":
        return "error"
    return "infer"


def _effective_validation_profile(
    validation_profile: ValidationProfile | None,
) -> ValidationProfile | None:
    if validation_profile == "lenient":
        return "data-light"
    return validation_profile


def _validation_context(
    *,
    table_schema: TableSchema,
    validation_profile: ValidationProfile | None,
) -> ColumnarValidationContext:
    return ColumnarValidationContext(
        table_schema=table_schema,
        schema_observation=None,
        validation_profile=_effective_validation_profile(validation_profile),
    )


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
    context = _validation_context(
        table_schema=table_schema,
        validation_profile=validation_profile,
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
                    contract_schema=contract_schema,
                )
                errors = list(read_errors)
                try:
                    validate_record_batch_reader(
                        table_key,
                        reader,
                        context=context,
                        mode="strict",
                    )
                except TableValidationError as exc:
                    errors.extend(exc.errors)
            else:
                validate_parquet_path(table_key, path, context=context, mode="strict")
                errors = []
        except TableValidationError as exc:
            errors = list(exc.errors)
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
