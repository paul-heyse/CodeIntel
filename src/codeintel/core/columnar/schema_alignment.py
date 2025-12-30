"""Arrow schema alignment helpers for ingest pipelines."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.schemas.contracts import (
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    EXTRAS_POLICIES,
    ExtrasPolicy,
    arrow_schema_from_fields,
)

_JSON_SEPARATORS = (",", ":")


@dataclass(frozen=True, slots=True)
class _AlignmentContext:
    """Shared alignment context for contract enforcement."""

    extras_policy: ExtrasPolicy
    extras_column: str
    extra_fields: set[str]
    promote_options: pc.CastOptions | None


def align_reader_to_contract(
    reader: pa.RecordBatchReader,
    contract_schema: pa.Schema,
    *,
    extras_policy: ExtrasPolicy,
    promote_options: pc.CastOptions | None = None,
) -> pa.RecordBatchReader:
    """Align a RecordBatchReader to a contract schema.

    Parameters
    ----------
    reader
        Incoming record batch reader to align.
    contract_schema
        Canonical Arrow schema contract to align to.
    extras_policy
        Policy for handling extra columns (retain, reject, drop).
    promote_options
        Optional casting options to allow explicit type promotion.

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
    _validate_extras_policy(extras_policy)
    extras_column = _extras_column_name(contract_schema)
    contract_names = {field.name for field in contract_schema}
    extra_fields = _extra_field_names(reader.schema, contract_names, extras_column)
    if extras_policy == "reject" and extra_fields:
        msg = f"Unexpected columns for contract: {sorted(extra_fields)}"
        raise ValueError(msg)
    context = _AlignmentContext(
        extras_policy=extras_policy,
        extras_column=extras_column,
        extra_fields=extra_fields,
        promote_options=promote_options,
    )
    target_schema = _target_schema(
        contract_schema=contract_schema,
        incoming_schema=reader.schema,
        context=context,
    )

    def _aligned_batches() -> Iterator[pa.RecordBatch]:
        for batch in reader:
            yield _align_batch(
                batch=batch,
                target_schema=target_schema,
                context=context,
            )

    return pa.RecordBatchReader.from_batches(target_schema, _aligned_batches())


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
    metadata = _decode_metadata(schema.metadata)
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
    if context.promote_options is not None:
        unified = pa.unify_schemas(
            [contract_schema, incoming_schema],
            promote_options=context.promote_options,
        )
        resolved_fields = [_resolved_field(field, unified) for field in contract_schema]
        base_schema = arrow_schema_from_fields(
            fields=resolved_fields,
            metadata=contract_schema.metadata,
        )
    if context.extras_column in base_schema.names:
        return base_schema
    if context.extras_policy != "retain" or not context.extra_fields:
        return base_schema
    return base_schema.append(_extras_field(context.extras_column))


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
            arrays.append(_coerce_array(field, extras_array, context.promote_options))
            continue
        if field.name in batch_schema.names:
            index = batch_schema.get_field_index(field.name)
            arrays.append(_coerce_array(field, batch.column(index), context.promote_options))
            continue
        arrays.append(pa.nulls(batch.num_rows, type=field.type))
    return pa.record_batch(arrays, schema=target_schema)


def _extras_array(batch: pa.RecordBatch, extra_fields: set[str]) -> pa.Array:
    extras_columns: dict[str, list[object]] = {}
    for name in sorted(extra_fields):
        index = batch.schema.get_field_index(name)
        extras_columns[name] = batch.column(index).to_pylist()
    payload: list[str | None] = []
    for row_index in range(batch.num_rows):
        row_payload = {name: values[row_index] for name, values in extras_columns.items()}
        if all(value is None for value in row_payload.values()):
            payload.append(None)
        else:
            payload.append(json.dumps(row_payload, sort_keys=True, separators=_JSON_SEPARATORS))
    return pa.array(payload, type=pa.string())


def _cast_array(
    array: pa.Array,
    target_type: pa.DataType,
    promote_options: pc.CastOptions | None,
) -> pa.Array:
    if array.type == target_type:
        return array
    if promote_options is None:
        return pc.cast(array, target_type)
    return pc.cast(array, target_type, options=promote_options)


def _coerce_array(
    field: pa.Field,
    array: pa.Array,
    promote_options: pc.CastOptions | None,
) -> pa.Array:
    try:
        return _cast_array(array, field.type, promote_options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        if _is_json_field(field) and pa.types.is_string(field.type):
            return _json_string_array(array)
        if pa.types.is_timestamp(field.type) and _is_string_array(array):
            parsed = _timestamp_string_array(array, field.type)
            if parsed is not None:
                return parsed
        raise


def _is_json_field(field: pa.Field) -> bool:
    metadata = _decode_metadata(field.metadata)
    return metadata.get("codeintel.column_type") == "JSON"


def _json_string_array(array: pa.Array) -> pa.Array:
    return pa.array(
        [_json_string_value(value) for value in array.to_pylist()],
        type=pa.string(),
    )


def _json_string_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=_JSON_SEPARATORS)


def _is_string_array(array: pa.Array) -> bool:
    return pa.types.is_string(array.type) or pa.types.is_large_string(array.type)


def _timestamp_string_array(
    array: pa.Array,
    target_type: pa.TimestampType,
) -> pa.Array | None:
    values: list[datetime | None] = []
    for raw in array.to_pylist():
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
    return pa.field(name, pa.string(), nullable=True)


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
    metadata = _decode_metadata(contract_schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return DEFAULT_EXTRAS_COLUMN


def _decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def _decode_metadata_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _validate_extras_policy(extras_policy: ExtrasPolicy) -> None:
    if extras_policy not in EXTRAS_POLICIES:
        msg = f"Unsupported extras policy: {extras_policy!r}"
        raise ValueError(msg)


__all__ = ["align_reader_to_contract", "extras_policy_from_schema"]
