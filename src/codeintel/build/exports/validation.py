"""Validate exported JSONL/Parquet datasets against JSON Schemas.

This module provides validation utilities for exported dataset files.
Validation uses generated JSON Schemas from the schema registry.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import jsonschema
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from referencing import Registry

from codeintel.build.errors import BuildProblemError
from codeintel.build.exports.common import log_export_error
from codeintel.build.schemas.json_schema_registry import (
    get_json_schema,
    get_json_schema_for_table_schema,
)
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.errors.schema import SCHEMA_NOT_FOUND, SCHEMA_VALIDATION_FAILED, SchemaError
from codeintel.core.errors.taxonomy import NOT_FOUND
from codeintel.core.schemas.primitives import column_type_base

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

log = logging.getLogger(__name__)
_arrow_compute_errors: list[type[BaseException]] = [TypeError, pa.ArrowInvalid]
_arrow_not_implemented = getattr(pa, "ArrowNotImplementedError", None)
if isinstance(_arrow_not_implemented, type):
    _arrow_compute_errors.append(_arrow_not_implemented)
_arrow_not_implemented_legacy = getattr(pa, "ArrowNotImplemented", None)
if isinstance(_arrow_not_implemented_legacy, type) and (
    _arrow_not_implemented_legacy is not _arrow_not_implemented
):
    _arrow_compute_errors.append(_arrow_not_implemented_legacy)
_ARROW_COMPUTE_ERRORS = tuple(_arrow_compute_errors)


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


def _get_inferred_schema(
    gateway: StorageGateway | None,
    table_key: str,
) -> TableSchema | None:
    if gateway is None:
        return None
    try:
        return gateway.schemas.load_table_schema(table_key)
    except (RuntimeError, TypeError, ValueError):
        return None


def _get_latest_observation(
    gateway: StorageGateway | None,
    table_key: str,
) -> SchemaObservationRecord | None:
    if gateway is None:
        return None
    try:
        return gateway.schemas.load_latest_schema_observation(table_key=table_key)
    except (RuntimeError, TypeError, ValueError):
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


@dataclass(frozen=True, slots=True)
class _ArrowFieldConstraint:
    name: str
    types: tuple[str, ...]
    required: bool
    nullable: bool
    enum: tuple[object, ...] | None
    minimum: float | int | None
    maximum: float | int | None
    min_length: int | None
    max_length: int | None


@dataclass(frozen=True, slots=True)
class _ParquetValidationOverrides:
    constraints: list[_ArrowFieldConstraint]
    additional_properties: bool
    needs_fallback: bool


_SUPPORTED_PROPERTY_KEYS = frozenset(
    {
        "type",
        "format",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "description",
        "$comment",
    }
)


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


def _schema_constraints(
    schema: dict[str, Any],
) -> tuple[list[_ArrowFieldConstraint], bool, bool]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return [], True, True
    required = set(schema.get("required") or ())
    additional_properties = bool(schema.get("additionalProperties", True))
    requires_fallback = _schema_needs_fallback(schema, properties=properties)
    constraints: list[_ArrowFieldConstraint] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            requires_fallback = True
            continue
        types = _normalize_types(prop.get("type"))
        nullable = "null" in types
        enum_values = prop.get("enum")
        enum = tuple(enum_values) if isinstance(enum_values, list) else None
        minimum = _number_or_none(prop.get("minimum"))
        maximum = _number_or_none(prop.get("maximum"))
        min_length = _int_or_none(prop.get("minLength"))
        max_length = _int_or_none(prop.get("maxLength"))
        constraints.append(
            _ArrowFieldConstraint(
                name=name,
                types=types,
                required=name in required,
                nullable=nullable,
                enum=enum,
                minimum=minimum,
                maximum=maximum,
                min_length=min_length,
                max_length=max_length,
            )
        )
    return constraints, additional_properties, requires_fallback


def _constraints_from_observation(
    table_schema: TableSchema,
    observation: SchemaObservationRecord,
) -> tuple[list[_ArrowFieldConstraint], bool]:
    constraints: list[_ArrowFieldConstraint] = []
    column_stats = observation.column_stats or {}
    extras_policy = None
    if observation.derived_settings is not None:
        extras_policy_raw = observation.derived_settings.get("extras_policy")
        extras_policy = extras_policy_raw if isinstance(extras_policy_raw, str) else None
    allow_extras = extras_policy != "reject"
    for column in table_schema.columns:
        stats = column_stats.get(column.name)
        min_value = None
        max_value = None
        if isinstance(stats, dict):
            min_value = _number_or_none(stats.get("min"))
            max_value = _number_or_none(stats.get("max"))
        constraints.append(
            _ArrowFieldConstraint(
                name=column.name,
                types=_json_types_for_column(column.type),
                required=not column.nullable,
                nullable=column.nullable,
                enum=None,
                minimum=min_value,
                maximum=max_value,
                min_length=None,
                max_length=None,
            )
        )
    return constraints, allow_extras


def _schema_needs_fallback(schema: dict[str, Any], *, properties: dict[str, Any]) -> bool:
    unsupported_root = set(schema) - {
        "$schema",
        "$id",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
    if unsupported_root:
        return True
    for prop in properties.values():
        if not isinstance(prop, dict):
            return True
        unsupported = set(prop) - _SUPPORTED_PROPERTY_KEYS
        if unsupported:
            return True
    return False


def _json_types_for_column(column_type: str) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "BOOLEAN": ("boolean",),
        "INTEGER": ("integer",),
        "BIGINT": ("integer",),
        "DOUBLE": ("number",),
        "DECIMAL": ("number",),
        "VARCHAR": ("string",),
        "JSON": ("object",),
        "TIMESTAMP": ("string",),
        "TIMESTAMPTZ": ("string",),
        "STRUCT": ("object",),
        "MAP": ("object",),
        "LIST": ("array",),
        "UNION": ("object",),
    }
    base = column_type_base(column_type)
    if base == "DECIMAL":
        compact = str(column_type).upper().replace(" ", "")
        match = re.match(r"^DECIMAL\\((\\d+),(\\d+)\\)$", compact)
        if match is not None and int(match.group(2)) == 0:
            return ("integer",)
    return mapping.get(base, ())


def _normalize_types(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def _number_or_none(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _schema_errors_for_json(
    schema: dict[str, Any],
    arrow_schema: pa.Schema,
    *,
    additional_properties: bool,
) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ["Schema properties missing or invalid"]
    expected_names = list(properties.keys())
    actual_names = list(arrow_schema.names)

    missing = [name for name in expected_names if name not in actual_names]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if not additional_properties:
        extra = [name for name in actual_names if name not in expected_names]
        if extra:
            errors.append(f"Unexpected columns: {', '.join(extra)}")
    return errors


def _validate_batch_constraints(
    batch: pa.RecordBatch,
    constraints: list[_ArrowFieldConstraint],
) -> list[str]:
    errors: list[str] = []
    names = set(batch.schema.names)
    for constraint in constraints:
        if constraint.name not in names:
            continue
        array = batch.column(batch.schema.get_field_index(constraint.name))
        if constraint.types and not _arrow_type_matches(constraint.types, array.type):
            errors.append(
                "Column "
                f"{constraint.name} type mismatch for JSON types {constraint.types}: {array.type}"
            )
            continue
        if constraint.required and not constraint.nullable and _has_nulls(array):
            errors.append(f"Column {constraint.name} contains nulls but is non-nullable")
        if constraint.enum is not None:
            errors.extend(_validate_enum(constraint, array))
        if constraint.minimum is not None or constraint.maximum is not None:
            errors.extend(_validate_range(constraint, array))
        if constraint.min_length is not None or constraint.max_length is not None:
            errors.extend(_validate_length(constraint, array))
    return errors


def _has_nulls(values: pa.Array | pa.ChunkedArray) -> bool:
    null_count = values.null_count
    if null_count is not None and null_count >= 0:
        return null_count > 0
    if isinstance(values, pa.ChunkedArray):
        return any((chunk.null_count or 0) > 0 for chunk in values.chunks)
    return False


def _arrow_type_matches(types: tuple[str, ...], data_type: pa.DataType) -> bool:
    normalized = data_type
    if pa.types.is_dictionary(normalized):
        normalized = normalized.value_type
    return any(_matches_json_type(json_type, normalized) for json_type in types)


def _matches_json_type(json_type: str, data_type: pa.DataType) -> bool:
    if json_type in {"object", "array"}:
        return True
    predicates: dict[str, Callable[[pa.DataType], bool]] = {
        "null": pa.types.is_null,
        "boolean": pa.types.is_boolean,
        "integer": lambda dtype: pa.types.is_integer(dtype) or pa.types.is_decimal(dtype),
        "number": lambda dtype: (
            pa.types.is_floating(dtype) or pa.types.is_integer(dtype) or pa.types.is_decimal(dtype)
        ),
        "string": lambda dtype: pa.types.is_string(dtype) or pa.types.is_large_string(dtype),
    }
    predicate = predicates.get(json_type)
    if predicate is None:
        return True
    return predicate(data_type)


def _validate_enum(
    constraint: _ArrowFieldConstraint,
    array: pa.Array | pa.ChunkedArray,
) -> list[str]:
    func = getattr(pc, "is_in", None)
    if not callable(func):
        return []
    try:
        mask = func(array, value_set=constraint.enum)
    except _ARROW_COMPUTE_ERRORS:
        return []
    mask = _apply_nullable_mask(mask, array, nullable=constraint.nullable)
    if _all_true(mask):
        return []
    failures = _first_false_indices(mask)
    return [
        f"Column {constraint.name} enum constraint failed at rows {failures}"
        if failures
        else f"Column {constraint.name} enum constraint failed"
    ]


def _validate_range(
    constraint: _ArrowFieldConstraint,
    array: pa.Array | pa.ChunkedArray,
) -> list[str]:
    mask: pa.Array | pa.ChunkedArray | None = None
    try:
        if constraint.minimum is not None:
            minimum_mask = _binary_compute("greater_equal", array, constraint.minimum)
            if minimum_mask is None:
                return []
            mask = minimum_mask if mask is None else _binary_compute("and_", mask, minimum_mask)
        if constraint.maximum is not None:
            maximum_mask = _binary_compute("less_equal", array, constraint.maximum)
            if maximum_mask is None:
                return []
            mask = maximum_mask if mask is None else _binary_compute("and_", mask, maximum_mask)
    except _ARROW_COMPUTE_ERRORS:
        return []
    if mask is None:
        return []
    mask = _apply_nullable_mask(mask, array, nullable=constraint.nullable)
    if _all_true(mask):
        return []
    failures = _first_false_indices(mask)
    return [
        f"Column {constraint.name} range constraint failed at rows {failures}"
        if failures
        else f"Column {constraint.name} range constraint failed"
    ]


def _validate_length(
    constraint: _ArrowFieldConstraint,
    array: pa.Array | pa.ChunkedArray,
) -> list[str]:
    errors: list[str] = []
    length_fn = getattr(pc, "utf8_length", None) or getattr(pc, "count_characters", None)
    lengths: pa.Array | pa.ChunkedArray | None = None
    if callable(length_fn):
        try:
            lengths = length_fn(array)
        except _ARROW_COMPUTE_ERRORS:
            lengths = None
    if lengths is None:
        return errors
    mask = _length_mask(constraint, lengths)
    if mask is None:
        return errors
    mask = _apply_nullable_mask(mask, array, nullable=constraint.nullable)
    if _all_true(mask):
        return errors
    failures = _first_false_indices(mask)
    errors.append(
        f"Column {constraint.name} length constraint failed at rows {failures}"
        if failures
        else f"Column {constraint.name} length constraint failed"
    )
    return errors


def _apply_nullable_mask(
    mask: pa.Array | pa.ChunkedArray,
    array: pa.Array | pa.ChunkedArray,
    *,
    nullable: bool,
) -> pa.Array | pa.ChunkedArray:
    if not nullable:
        return mask
    nulls = _unary_compute("is_null", array)
    if nulls is None:
        return mask
    combined = _binary_compute("or_", nulls, mask)
    return combined if combined is not None else mask


def _length_mask(
    constraint: _ArrowFieldConstraint,
    lengths: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray | None:
    mask: pa.Array | pa.ChunkedArray | None = None
    if constraint.min_length is not None:
        min_mask = _binary_compute("greater_equal", lengths, constraint.min_length)
        if min_mask is None:
            return None
        mask = min_mask if mask is None else _binary_compute("and_", mask, min_mask)
        if mask is None:
            return None
    if constraint.max_length is not None:
        max_mask = _binary_compute("less_equal", lengths, constraint.max_length)
        if max_mask is None:
            return None
        mask = max_mask if mask is None else _binary_compute("and_", mask, max_mask)
        if mask is None:
            return None
    return mask


def _all_true(mask: pa.Array | pa.ChunkedArray) -> bool:
    all_fn = getattr(pc, "all", None)
    if callable(all_fn):
        try:
            result = all_fn(mask)
            as_py = getattr(result, "as_py", None)
            if callable(as_py):
                value = as_py()
                return bool(value) if value is not None else False
        except _ARROW_COMPUTE_ERRORS:
            pass
    try:
        return all(mask.to_pylist())
    except _ARROW_COMPUTE_ERRORS:
        return False


def _first_false_indices(mask: pa.Array | pa.ChunkedArray, *, limit: int = 5) -> list[int]:
    try:
        values = mask.to_pylist()
    except _ARROW_COMPUTE_ERRORS:
        return []
    failures = [idx for idx, ok in enumerate(values) if not ok]
    return failures[:limit]


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


def _validate_parquet(
    path: Path,
    *,
    schema: dict[str, Any],
    validator: jsonschema.Draft202012Validator,
    overrides: _ParquetValidationOverrides | None = None,
) -> list[str]:
    """Validate a Parquet file against a schema.

    Parameters
    ----------
    path
        Path to Parquet file.
    schema
        JSON Schema used for validation.
    validator
        JSON Schema validator instance.
    overrides
        Optional overrides for constraints and extras handling.

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    dataset = ds.dataset(str(path), format="parquet")
    errors: list[str] = []
    if overrides is None:
        constraints, additional_properties, needs_fallback = _schema_constraints(schema)
    else:
        constraints = overrides.constraints
        additional_properties = overrides.additional_properties
        needs_fallback = overrides.needs_fallback
    errors.extend(
        _schema_errors_for_json(
            schema,
            dataset.schema,
            additional_properties=additional_properties,
        )
    )
    reader = dataset.scanner().to_reader()
    for batch in reader:
        errors.extend(_validate_batch_constraints(batch, constraints))
        if needs_fallback:
            records = _records_from_batch(batch)
            errors.extend(_validate_records(records, validator))
    return errors


def _binary_compute(
    name: str,
    left: pa.Array | pa.ChunkedArray,
    right: object,
) -> pa.Array | pa.ChunkedArray | None:
    name = _normalize_compute_name(name)
    func = getattr(pc, name, None)
    if not callable(func):
        return None
    try:
        return func(left, right)
    except _ARROW_COMPUTE_ERRORS:
        return None


def _unary_compute(
    name: str,
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray | None:
    func = getattr(pc, name, None)
    if not callable(func):
        return None
    try:
        return func(array)
    except _ARROW_COMPUTE_ERRORS:
        return None


def _normalize_compute_name(name: str) -> str:
    if name == "and_":
        return "and"
    if name == "or_":
        return "or"
    return name


def _records_from_batch(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    columns = batch.schema.names
    arrays = [batch.column(idx) for idx in range(batch.num_columns)]
    return [
        {name: arrays[idx][row_idx].as_py() for idx, name in enumerate(columns)}
        for row_idx in range(batch.num_rows)
    ]


def validate_export_files(
    table_key: str,
    paths: list[Path],
    *,
    dataset_name: str | None = None,
    gateway: StorageGateway | None = None,
) -> int:
    """Validate files against the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    dataset_name
        Optional dataset name used for logging context.
    gateway
        Optional storage gateway for inference-backed schema observations.
    paths
        List of JSONL or Parquet files to validate.
    gateway
        Optional storage gateway used to load inferred schemas.

    Returns
    -------
    int
        0 on success, non-zero on validation failure.

    Raises
    ------
    BuildProblemError
        If no schema is available for the table.
    """
    inferred_schema = _get_inferred_schema(gateway, table_key)
    observation = _get_latest_observation(gateway, table_key)
    schema = (
        get_json_schema_for_table_schema(inferred_schema)
        if inferred_schema is not None
        else _get_generated_schema(table_key)
    )
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

    overrides: _ParquetValidationOverrides | None = None
    if inferred_schema is not None and observation is not None:
        constraints_override, allow_extras_override = _constraints_from_observation(
            inferred_schema, observation
        )
        overrides = _ParquetValidationOverrides(
            constraints=constraints_override,
            additional_properties=(
                allow_extras_override if allow_extras_override is not None else True
            ),
            needs_fallback=False,
        )

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
            errors = _validate_parquet(
                path,
                schema=schema,
                validator=validator,
                overrides=overrides,
            )
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
