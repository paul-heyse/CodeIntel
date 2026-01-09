"""Arrow schema alignment helpers for ingest pipelines."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import msgspec
import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.conversion import record_batch_reader_from_iterable, table_to_reader
from codeintel.core.columnar.nested_ops import deep_cast_array
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.columnar.schema_ops import unify_schemas
from codeintel.core.columnar.type_normalization import binary_view_cast_type, string_view_cast_type
from codeintel.core.schemas.arrow_gen import (
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    EXTRAS_POLICIES,
    ExtrasPolicy,
)
from codeintel.core.serialization.payload import encode_payload

_JSON_ENCODER = msgspec.json.Encoder(order="deterministic")


@dataclass(frozen=True, slots=True)
class _AlignmentContext:
    """Shared alignment context for contract enforcement."""

    extras_policy: ExtrasPolicy
    extras_column: str
    extra_fields: set[str]
    schema_promote_options: SchemaPromoteOptions
    cast_options: pc.CastOptions | None


def align_reader_to_contract(
    reader: pa.RecordBatchReader,
    contract_schema: pa.Schema,
    *,
    extras_policy: ExtrasPolicy | None = None,
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
    cast_options: pc.CastOptions | None = None,
) -> pa.RecordBatchReader:
    """Align a RecordBatchReader to a contract schema.

    Parameters
    ----------
    reader
        Incoming record batch reader to align.
    contract_schema
        Canonical Arrow schema contract to align to.
    extras_policy
        Policy for handling extra columns (retain, reject, drop). When None,
        resolve from Arrow schema metadata.
    schema_promote_options
        Promotion policy for schema unification.
    cast_options
        Optional cast options to allow explicit type promotion.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader that yields batches aligned to the contract schema.

    Raises
    ------
    ValueError
        If extras_policy is invalid or unexpected columns are present under
        a reject policy.
    """
    normalized_contract = _normalize_view_schema(contract_schema)
    normalized_incoming = _normalize_view_schema(reader.schema)
    resolved_policy = extras_policy or extras_policy_from_schema(normalized_contract)
    _validate_extras_policy(resolved_policy)
    extras_column = _extras_column_name(normalized_contract)
    contract_names = {field.name for field in normalized_contract}
    extra_fields = _extra_field_names(normalized_incoming, contract_names, extras_column)
    if resolved_policy == "reject" and extra_fields:
        msg = f"Unexpected columns for contract: {sorted(extra_fields)}"
        raise ValueError(msg)
    context = _AlignmentContext(
        extras_policy=resolved_policy,
        extras_column=extras_column,
        extra_fields=extra_fields,
        schema_promote_options=schema_promote_options,
        cast_options=cast_options,
    )
    target_schema = _target_schema(
        contract_schema=normalized_contract,
        incoming_schema=normalized_incoming,
        context=context,
    )

    def _aligned_batches() -> Iterator[pa.RecordBatch]:
        for batch in reader:
            yield _align_batch(
                batch=batch,
                target_schema=target_schema,
                context=context,
            )

    aligned = record_batch_reader_from_iterable(_aligned_batches(), empty_policy="none")
    if aligned is None:
        return empty_reader_from_schema(target_schema)
    return aligned


def align_table_to_contract(
    table: pa.Table,
    contract_schema: pa.Schema,
    *,
    extras_policy: ExtrasPolicy | None = None,
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
    cast_options: pc.CastOptions | None = None,
) -> pa.Table:
    """Align an Arrow table to a contract schema.

    Parameters
    ----------
    table
        Incoming Arrow table to align.
    contract_schema
        Canonical Arrow schema contract to align to.
    extras_policy
        Policy for handling extra columns (retain, reject, drop). When None,
        resolve from Arrow schema metadata.
    schema_promote_options
        Promotion policy for schema unification.
    cast_options
        Optional cast options to allow explicit type promotion.

    Returns
    -------
    pa.Table
        Table aligned to the contract schema.
    """
    reader = table_to_reader(table, batch_size=None)
    aligned = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy,
        schema_promote_options=schema_promote_options,
        cast_options=cast_options,
    )
    return pa.Table.from_batches(aligned, schema=aligned.schema)


def extras_policy_from_schema(
    schema: pa.Schema,
    *,
    default: ExtrasPolicy = DEFAULT_EXTRAS_POLICY,
) -> ExtrasPolicy:
    """Resolve the extras policy from Arrow schema metadata.

    Parameters
    ----------
    schema
        Arrow schema with optional contract metadata.
    default
        Fallback extras policy when metadata is missing or invalid.

    Returns
    -------
    ExtrasPolicy
        Resolved extras policy.
    """
    metadata = decode_metadata(schema.metadata)
    raw = metadata.get("codeintel.extras_policy")
    if isinstance(raw, str) and raw in EXTRAS_POLICIES:
        return cast("ExtrasPolicy", raw)
    return default


def _target_schema(
    *,
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
    context: _AlignmentContext,
) -> pa.Schema:
    base_schema = contract_schema
    unified = _unify_schemas(
        [contract_schema, incoming_schema],
        promote_options=context.schema_promote_options,
    )
    resolved_fields = [_resolved_field(field, unified) for field in contract_schema]
    base_schema = pa.schema(resolved_fields, metadata=contract_schema.metadata)
    if context.extras_column in base_schema.names:
        return base_schema
    if context.extras_policy != "retain" or not context.extra_fields:
        return base_schema
    return base_schema.append(_extras_field(context.extras_column))


def _normalize_view_schema(schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    changed = False
    for field in schema:
        normalized_type = _normalize_view_type(field.type)
        if normalized_type != field.type:
            updated_field = field.with_type(normalized_type)
            changed = True
        else:
            updated_field = field
        fields.append(updated_field)
    if not changed:
        return schema
    return pa.schema(fields, metadata=schema.metadata)


def _normalize_view_type(data_type: pa.DataType) -> pa.DataType:
    normalized = string_view_cast_type(data_type)
    return binary_view_cast_type(normalized)


def _resolved_field(field: pa.Field, unified: pa.Schema) -> pa.Field:
    try:
        unified_field = unified.field(field.name)
    except KeyError:
        return field
    return unified_field.with_metadata(field.metadata)


def _align_batch(
    *,
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
    context: _AlignmentContext,
) -> pa.RecordBatch:
    arrays: list[pa.Array] = []
    batch_schema = batch.schema
    extras_array: pa.Array | None = None
    if context.extras_policy == "retain" and context.extra_fields:
        extras_array = _extras_array(batch, context.extra_fields)
    for field in target_schema:
        if field.name == context.extras_column and extras_array is not None:
            arrays.append(_coerce_array(field, extras_array, context.cast_options))
            continue
        if field.name in batch_schema.names:
            index = batch_schema.get_field_index(field.name)
            arrays.append(_coerce_array(field, batch.column(index), context.cast_options))
            continue
        arrays.append(pa.nulls(batch.num_rows, type=field.type))
    return pa.record_batch(arrays, schema=target_schema)


def _extras_array(batch: pa.RecordBatch, extra_fields: set[str]) -> pa.Array:
    extras_columns: dict[str, list[object]] = {}
    for name in sorted(extra_fields):
        index = batch.schema.get_field_index(name)
        extras_columns[name] = _array_values(batch.column(index))
    payload: list[bytes | None] = []
    for row_index in range(batch.num_rows):
        row_payload = {name: values[row_index] for name, values in extras_columns.items()}
        if all(value is None for value in row_payload.values()):
            payload.append(None)
        else:
            payload.append(encode_payload(row_payload))
    return pa.array(payload, type=pa.binary())


def _cast_array(
    array: pa.Array,
    target_type: pa.DataType,
    cast_options: pc.CastOptions | None,
) -> pa.Array:
    if array.type == target_type:
        return array
    if cast_options is None:
        return pc.cast(array, target_type)
    return pc.cast(array, target_type, options=cast_options)


def _coerce_array(
    field: pa.Field,
    array: pa.Array,
    cast_options: pc.CastOptions | None,
) -> pa.Array:
    try:
        if _is_nested_type(field.type):
            return deep_cast_array(array, field.type, cast_options=cast_options)
        return _cast_array(array, field.type, cast_options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        if _is_json_field(field) and pa.types.is_string(field.type):
            return _json_string_array(array)
        if pa.types.is_timestamp(field.type) and _is_string_array(array):
            parsed = _timestamp_string_array(array, field.type)
            if parsed is not None:
                return parsed
        raise


def _unify_schemas(
    schemas: list[pa.Schema],
    *,
    promote_options: SchemaPromoteOptions,
) -> pa.Schema:
    return unify_schemas(schemas, promote_options=promote_options)


def _is_json_field(field: pa.Field) -> bool:
    metadata = decode_metadata(field.metadata)
    return metadata.get("codeintel.column_type") == "JSON"


def _json_string_array(array: pa.Array) -> pa.Array:
    return pa.array(
        [_json_string_value(_scalar_py(value)) for value in array],
        type=pa.string(),
    )


def _json_string_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _JSON_ENCODER.encode(value).decode("utf-8")


def _is_string_array(array: pa.Array) -> bool:
    return pa.types.is_string(array.type) or pa.types.is_large_string(array.type)


def _timestamp_string_array(
    array: pa.Array,
    target_type: pa.TimestampType,
) -> pa.Array | None:
    values: list[datetime | None] = []
    for value in array:
        raw = _scalar_py(value)
        if raw is None:
            values.append(None)
            continue
        if not isinstance(raw, str):
            return None
        parsed = _parse_iso_timestamp(raw, target_type)
        if parsed is None:
            return None
        values.append(parsed)
    return pa.array(values, type=target_type)


def _scalar_py(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return as_py()
    return value


def _array_values(array: pa.Array) -> list[object]:
    return [_scalar_py(value) for value in array]


def _parse_iso_timestamp(value: str, target_type: pa.TimestampType) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if target_type.tz is None:
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _extras_field(name: str) -> pa.Field:
    return pa.field(name, pa.binary(), nullable=True)


def _extra_field_names(
    incoming_schema: pa.Schema,
    contract_names: set[str],
    extras_column: str,
) -> set[str]:
    return {
        name
        for name in incoming_schema.names
        if name not in contract_names and name != extras_column
    }


def _extras_column_name(contract_schema: pa.Schema) -> str:
    metadata = decode_metadata(contract_schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return DEFAULT_EXTRAS_COLUMN


def _validate_extras_policy(extras_policy: ExtrasPolicy) -> None:
    if extras_policy not in EXTRAS_POLICIES:
        msg = f"Unsupported extras policy: {extras_policy!r}"
        raise ValueError(msg)


def _is_nested_type(data_type: pa.DataType) -> bool:
    return pa.types.is_struct(data_type) or pa.types.is_map(data_type) or _is_list_type(data_type)


def _is_list_type(data_type: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    list_view = bool(callable(is_list_view) and is_list_view(data_type))
    large_list_view = bool(callable(is_large_list_view) and is_large_list_view(data_type))
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
        or list_view
        or large_list_view
    )


__all__ = ["align_reader_to_contract", "extras_policy_from_schema"]
