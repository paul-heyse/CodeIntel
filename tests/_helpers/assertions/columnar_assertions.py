"""Columnar contract assertion helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.schema_alignment import extras_policy_from_schema
from codeintel.core.schemas.arrow_gen import DEFAULT_EXTRAS_COLUMN, arrow_contract_for_table_schema
from codeintel.storage.contracts.schema_provider import get_schema_provider

if TYPE_CHECKING:
    from codeintel.core.columnar.schema_alignment import ExtrasPolicy
    from codeintel.core.schemas.primitives import TableSchema


def assert_reader_matches_contract(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    table_schema: TableSchema | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> None:
    """Assert a RecordBatchReader schema matches the Arrow contract.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    reader
        RecordBatchReader to validate.
    table_schema
        Optional TableSchema override.
    extras_policy
        Optional extras policy override.
    """
    contract_schema, policy = _resolve_contract(table_key, table_schema, extras_policy)
    _ensure_contract_alignment(table_key, reader.schema, contract_schema, policy)


def assert_table_matches_contract(
    table_key: str,
    table: pa.Table,
    *,
    table_schema: TableSchema | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> None:
    """Assert an Arrow table schema matches the Arrow contract.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    table
        Arrow table to validate.
    table_schema
        Optional TableSchema override.
    extras_policy
        Optional extras policy override.
    """
    contract_schema, policy = _resolve_contract(table_key, table_schema, extras_policy)
    _ensure_contract_alignment(table_key, table.schema, contract_schema, policy)


def _resolve_contract(
    table_key: str,
    table_schema: TableSchema | None,
    extras_policy: ExtrasPolicy | None,
) -> tuple[pa.Schema, ExtrasPolicy]:
    schema = table_schema or _lookup_table_schema(table_key)
    contract_schema = arrow_contract_for_table_schema(table_schema=schema)
    policy = extras_policy or extras_policy_from_schema(contract_schema)
    return contract_schema, policy


def _lookup_table_schema(table_key: str) -> TableSchema:
    provider = get_schema_provider()
    table_schema = provider.get_table_schema(table_key)
    if table_schema is None:
        message = f"No schema registered for {table_key}"
        raise AssertionError(message)
    return table_schema


def _ensure_contract_alignment(
    table_key: str,
    actual_schema: pa.Schema,
    contract_schema: pa.Schema,
    extras_policy: ExtrasPolicy,
) -> None:
    _ensure_schema_metadata(actual_schema, contract_schema)
    _ensure_schema_fields(table_key, actual_schema, contract_schema, extras_policy)


def _ensure_schema_metadata(actual_schema: pa.Schema, contract_schema: pa.Schema) -> None:
    contract_metadata = contract_schema.metadata or {}
    if not contract_metadata:
        return
    actual_metadata = actual_schema.metadata or {}
    for key, value in contract_metadata.items():
        if actual_metadata.get(key) != value:
            msg = f"Contract metadata mismatch for {key!r}"
            raise AssertionError(msg)


def _ensure_schema_fields(
    table_key: str,
    actual_schema: pa.Schema,
    contract_schema: pa.Schema,
    extras_policy: ExtrasPolicy,
) -> None:
    expected_names = list(contract_schema.names)
    if extras_policy == "retain":
        extras_column = _extras_column_from_schema(contract_schema)
        if extras_column in actual_schema.names and extras_column not in expected_names:
            expected_names = [*expected_names, extras_column]
    if actual_schema.names != expected_names:
        msg = (
            f"Contract schema mismatch for {table_key}: expected {expected_names}, "
            f"got {list(actual_schema.names)}"
        )
        raise AssertionError(msg)
    for field in contract_schema:
        actual = actual_schema.field(field.name)
        if actual.type != field.type:
            msg = (
                f"Contract type mismatch for {table_key}.{field.name}: "
                f"expected {field.type}, got {actual.type}"
            )
            raise AssertionError(msg)


def _extras_column_from_schema(schema: pa.Schema, *, default: str = DEFAULT_EXTRAS_COLUMN) -> str:
    metadata = _decode_metadata(schema.metadata)
    raw = metadata.get("codeintel.extras_column")
    if isinstance(raw, str) and raw:
        return raw
    return default


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, object]:
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


__all__ = ["assert_reader_matches_contract", "assert_table_matches_contract"]
