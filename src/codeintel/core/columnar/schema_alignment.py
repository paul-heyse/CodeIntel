"""Arrow schema alignment helpers for ingest pipelines."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

import pyarrow as pa

from codeintel.core.schemas.arrow_gen import (
    DEFAULT_EXTRAS_COLUMN,
    EXTRAS_POLICIES,
    ExtrasPolicy,
)

_JSON_SEPARATORS = (",", ":")


def align_reader_to_contract(
    reader: pa.RecordBatchReader,
    contract_schema: pa.Schema,
    *,
    extras_policy: ExtrasPolicy,
    promote_options: pa.compute.CastOptions | None = None,
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
    target_schema = _target_schema(
        contract_schema=contract_schema,
        incoming_schema=reader.schema,
        extras_policy=extras_policy,
        extras_column=extras_column,
        extra_fields=extra_fields,
        promote_options=promote_options,
    )

    def _aligned_batches() -> Iterator[pa.RecordBatch]:
        for batch in reader:
            yield _align_batch(
                batch=batch,
                target_schema=target_schema,
                contract_names=contract_names,
                extras_policy=extras_policy,
                extras_column=extras_column,
                extra_fields=extra_fields,
                promote_options=promote_options,
            )

    return pa.RecordBatchReader.from_batches(target_schema, _aligned_batches())


def _target_schema(
    *,
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
    extras_policy: ExtrasPolicy,
    extras_column: str,
    extra_fields: set[str],
    promote_options: pa.compute.CastOptions | None,
) -> pa.Schema:
    base_schema = contract_schema
    if promote_options is not None:
        unified = pa.unify_schemas(
            [contract_schema, incoming_schema],
            promote_options=promote_options,
        )
        resolved_fields = [_resolved_field(field, unified) for field in contract_schema]
        base_schema = pa.schema(resolved_fields, metadata=contract_schema.metadata)
    if extras_column in base_schema.names:
        return base_schema
    if extras_policy != "retain" or not extra_fields:
        return base_schema
    return base_schema.append(_extras_field(extras_column))


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
    contract_names: set[str],
    extras_policy: ExtrasPolicy,
    extras_column: str,
    extra_fields: set[str],
    promote_options: pa.compute.CastOptions | None,
) -> pa.RecordBatch:
    arrays: list[pa.Array] = []
    batch_schema = batch.schema
    extras_array: pa.Array | None = None
    if extras_policy == "retain" and extra_fields:
        extras_array = _extras_array(batch, extra_fields)
    for field in target_schema:
        if field.name == extras_column and extras_array is not None:
            arrays.append(_cast_array(extras_array, field.type, promote_options))
            continue
        if field.name in batch_schema.names:
            index = batch_schema.get_field_index(field.name)
            arrays.append(_cast_array(batch.column(index), field.type, promote_options))
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
    promote_options: pa.compute.CastOptions | None,
) -> pa.Array:
    if array.type == target_type:
        return array
    if promote_options is None:
        return pa.compute.cast(array, target_type)
    return pa.compute.cast(array, target_type, options=promote_options)


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


__all__ = ["align_reader_to_contract"]
