"""Shared schema constraint validation helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from codeintel.core.columnar.schema_alignment import extras_policy_from_schema
from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.schemas.arrow_gen import (
    ARROW_SCHEMA_METADATA_KEYS,
    DEFAULT_EXTRAS_COLUMN,
    EXTRAS_POLICIES,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.validation.arrow_type_compat import is_compatible_arrow_type
from codeintel.core.validation.profiles import (
    ValidationProfile,
    normalize_validation_profile,
    resolve_validation_depth,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord


def schema_errors(
    table_schema: TableSchema,
    actual_schema: pa.Schema,
    *,
    extras_column: str | None = None,
) -> list[str]:
    """Validate Arrow schema structure against a TableSchema.

    Returns
    -------
    list[str]
        Human-readable schema validation errors.
    """
    resolved_extras = extras_column or _extras_column_name(actual_schema)
    expected_names = [column.name for column in table_schema.columns]
    actual_names = [name for name in actual_schema.names if name != resolved_extras]
    allow_extra = extras_policy_from_schema(actual_schema) != "reject"
    expected_set = set(expected_names)

    errors: list[str] = []
    missing = [name for name in expected_names if name not in actual_names]
    extra = [name for name in actual_names if name not in expected_set]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if extra and not allow_extra:
        errors.append(f"Unexpected columns: {', '.join(extra)}")
    if not missing:
        ordered_actual = (
            [name for name in actual_names if name in expected_set] if allow_extra else actual_names
        )
        if expected_names != ordered_actual:
            errors.append(
                "Column order mismatch: expected "
                f"{', '.join(expected_names)} but got {', '.join(ordered_actual)}"
            )

    by_name = {column.name: column for column in table_schema.columns}
    for name in expected_names:
        if name not in actual_schema.names:
            continue
        column = by_name[name]
        actual_field = actual_schema.field(name)
        if not is_compatible_arrow_type(column, actual_field.type):
            errors.append(
                f"Column {name} type mismatch: expected {column.type}, got {actual_field.type}"
            )
    return errors


def arrow_table_errors(table: pa.Table) -> list[str]:
    """Return Arrow validation errors for a table.

    Returns
    -------
    list[str]
        Validation error messages for the table, if any.
    """
    try:
        table.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def arrow_batch_errors(batch: pa.RecordBatch) -> list[str]:
    """Return Arrow validation errors for a record batch.

    Returns
    -------
    list[str]
        Validation error messages for the record batch, if any.
    """
    try:
        batch.validate()
    except pa.ArrowInvalid as exc:
        return [f"Arrow validation failed: {exc}"]
    return []


def nullability_errors_for_table(table_schema: TableSchema, table: pa.Table) -> list[str]:
    """Validate non-nullable columns for an Arrow table.

    Returns
    -------
    list[str]
        Nullability validation errors for the table, if any.
    """
    errors: list[str] = []
    for column in table_schema.columns:
        if column.nullable:
            continue
        if column.name not in table.column_names:
            continue
        if not _all_valid(table.column(column.name)):
            errors.append(f"Column {column.name} contains nulls but is non-nullable")
    return errors


def nullability_errors_for_batch(
    table_schema: TableSchema,
    batch: pa.RecordBatch,
) -> list[str]:
    """Validate non-nullable columns for a record batch.

    Returns
    -------
    list[str]
        Nullability validation errors for the record batch, if any.
    """
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


def observation_errors_for_table(
    table_schema: TableSchema,
    table: pa.Table,
    observation: SchemaObservationRecord | None,
) -> list[str]:
    """Validate a table against observation-derived ranges.

    Returns
    -------
    list[str]
        Range validation errors based on the observation.
    """
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


def observation_errors_for_batch(
    table_schema: TableSchema,
    batch: pa.RecordBatch,
    observation: SchemaObservationRecord | None,
) -> list[str]:
    """Validate a record batch against observation-derived ranges.

    Returns
    -------
    list[str]
        Range validation errors based on the observation.
    """
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


def validate_parquet_path(
    path: Path,
    *,
    table_schema: TableSchema,
    observation: SchemaObservationRecord | None = None,
    validation_profile: ValidationProfile | None = None,
) -> list[str]:
    """Validate a Parquet file against schema constraints.

    Returns
    -------
    list[str]
        Validation errors accumulated across all batches.
    """
    parquet_file = pq.ParquetFile(path)
    errors = schema_errors(table_schema, parquet_file.schema_arrow)
    include_data_checks = True
    if validation_profile is not None:
        normalized = normalize_validation_profile(validation_profile, default="strict")
        include_data_checks = resolve_validation_depth(normalized) != "schema-only"
    for batch in parquet_file.iter_batches():
        errors.extend(arrow_batch_errors(batch))
        if include_data_checks:
            errors.extend(nullability_errors_for_batch(table_schema, batch))
            errors.extend(observation_errors_for_batch(table_schema, batch, observation))
    return errors


def iter_reader_batch_errors(
    reader: pa.RecordBatchReader,
    *,
    table_schema: TableSchema,
    observation: SchemaObservationRecord | None = None,
) -> Iterator[tuple[int, list[str]]]:
    """Yield per-batch validation errors for a reader.

    Yields
    ------
    tuple[int, list[str]]
        Pairs of batch index and validation errors.
    """
    for batch_index, batch in enumerate(reader):
        errors: list[str] = []
        errors.extend(arrow_batch_errors(batch))
        errors.extend(nullability_errors_for_batch(table_schema, batch))
        errors.extend(observation_errors_for_batch(table_schema, batch, observation))
        yield batch_index, errors


def schema_metadata_errors(schema: pa.Schema, *, allow_unknown_keys: bool = False) -> list[str]:
    """Return schema metadata validation errors for Arrow schemas.

    Parameters
    ----------
    schema
        Arrow schema to inspect.
    allow_unknown_keys
        When False, emit errors for unknown ``codeintel.*`` metadata keys.

    Returns
    -------
    list[str]
        Human-readable metadata validation errors.
    """
    metadata = decode_metadata(schema.metadata)
    errors: list[str] = []
    if not allow_unknown_keys:
        errors.extend(_unknown_schema_metadata_errors(metadata))
    errors.extend(_schema_metadata_type_errors(metadata))
    return errors


def _unknown_schema_metadata_errors(metadata: Mapping[str, object]) -> list[str]:
    return [
        f"Unknown schema metadata key: {key}"
        for key in metadata
        if key.startswith("codeintel.") and key not in ARROW_SCHEMA_METADATA_KEYS
    ]


def _schema_metadata_type_errors(metadata: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _string_metadata_errors(
            metadata,
            (
                "codeintel.table_key",
                "codeintel.schema_hash",
                "codeintel.schema_digest",
                "codeintel.schema_contract_version",
                "codeintel.extras_column",
                "codeintel.description",
            ),
        )
    )
    extras_policy = metadata.get("codeintel.extras_policy")
    if extras_policy is not None:
        if not isinstance(extras_policy, str):
            errors.append(
                "Arrow schema metadata codeintel.extras_policy must be a string, "
                f"got {type(extras_policy)}"
            )
        elif extras_policy not in EXTRAS_POLICIES:
            errors.append(
                "Arrow schema metadata codeintel.extras_policy must be one of "
                f"{sorted(EXTRAS_POLICIES)}, got {extras_policy!r}"
            )
    primary_key = metadata.get("codeintel.primary_key")
    if primary_key is not None and not _is_str_sequence(primary_key):
        errors.append(
            "Arrow schema metadata codeintel.primary_key must be a sequence of strings, "
            f"got {type(primary_key)}"
        )
    errors.extend(
        _mapping_metadata_errors(
            metadata,
            (
                "codeintel.extras_schema",
                "codeintel.provenance",
            ),
        )
    )
    return errors


def _string_metadata_errors(
    metadata: Mapping[str, object],
    keys: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            continue
        errors.append(f"Arrow schema metadata {key} must be a string, got {type(value)}")
    return errors


def _mapping_metadata_errors(
    metadata: Mapping[str, object],
    keys: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if _is_str_mapping(value):
            continue
        errors.append(
            f"Arrow schema metadata {key} must be a mapping with string keys, got {type(value)}"
        )
    return errors


def _is_str_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def _is_str_mapping(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(key, str) for key in value)


def _column_stats_lookup(
    observation: SchemaObservationRecord | None,
) -> Mapping[str, Mapping[str, object]]:
    if observation is None:
        return {}
    stats = observation.column_stats
    if isinstance(stats, Mapping):
        return stats
    return {}


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
        try:
            result = any_fn(values)
        except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            result = None
        if result is not None:
            as_py = getattr(result, "as_py", None)
            if callable(as_py):
                return bool(as_py() or False)
    if isinstance(values, pa.ChunkedArray):
        return any(_any_true(chunk) for chunk in values.chunks)
    try:
        for value in values:
            as_py = getattr(value, "as_py", None)
            resolved = as_py() if callable(as_py) else value
            if resolved:
                return True
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False
    else:
        return False


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


def _has_nulls(values: pa.Array | pa.ChunkedArray) -> bool:
    null_count = values.null_count
    if null_count is not None and null_count >= 0:
        return null_count > 0
    if isinstance(values, pa.ChunkedArray):
        return any((chunk.null_count or 0) > 0 for chunk in values.chunks)
    return False


def _coerce_scalar(bound: object, data_type: pa.DataType) -> pa.Scalar | None:
    try:
        return pa.scalar(bound, type=data_type)
    except (TypeError, pa.ArrowInvalid):
        return None


def _extras_column_name(schema: pa.Schema) -> str:
    metadata = decode_metadata(schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return DEFAULT_EXTRAS_COLUMN


__all__ = [
    "arrow_batch_errors",
    "arrow_table_errors",
    "iter_reader_batch_errors",
    "nullability_errors_for_batch",
    "nullability_errors_for_table",
    "observation_errors_for_batch",
    "observation_errors_for_table",
    "schema_errors",
    "validate_parquet_path",
]
