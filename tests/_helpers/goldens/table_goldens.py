"""Golden helpers for table dumps."""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.conversion import reader_to_table, tabular_to_arrow_table
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def dump_table(
    gateway: StorageGateway,
    table_key: str,
) -> pa.Table:
    """Load a table into an Arrow table for golden comparisons.

    Parameters
    ----------
    gateway
        Storage gateway to query.
    table_key
        Schema-qualified table key.

    Returns
    -------
    pyarrow.Table
        Arrow table for the requested dataset.

    Raises
    ------
    ValueError
        If the table key contains unsafe identifiers.
    """
    schema, table = split_table_key(table_key)
    if not _is_safe_identifier(schema) or not _is_safe_identifier(table):
        message = f"Unsafe identifier in table_key: {table_key!r}"
        raise ValueError(message)
    relation = gateway.con.table(f"{schema}.{table}")
    return tabular_to_arrow_table(relation, batch_size=DEFAULT_ARROW_BATCH_SIZE)


def assert_table_matches_golden(
    gateway: StorageGateway,
    table_key: str,
    *,
    golden_path: Path,
    update_mode: bool | None = None,
) -> pa.Table:
    """Assert a table dump matches a golden IPC file.

    Parameters
    ----------
    gateway
        Storage gateway to query.
    table_key
        Schema-qualified table key.
    golden_path
        Expected golden IPC file path.
    update_mode
        When True, overwrite golden files with current output.

    Returns
    -------
    pyarrow.Table
        Table payload used for the comparison.

    Raises
    ------
    AssertionError
        If the golden file is missing or differs from current output.
    """
    actual_table = dump_table(gateway, table_key)
    actual_payload = _ipc_bytes_for_table(actual_table)
    should_update = update_mode if update_mode is not None else _update_enabled()

    if should_update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_bytes(actual_payload)
        return actual_table

    if not golden_path.exists():
        message = (
            f"Golden file not found: {golden_path}\n"
            "Run with UPDATE_GOLDEN=1 to create it.\n"
            f"Schema:\n{_format_schema(actual_table.schema)}"
        )
        raise AssertionError(message)

    expected_table = _table_from_ipc(golden_path)
    if not actual_table.schema.equals(expected_table.schema, check_metadata=True):
        message = (
            f"Schema mismatch for {table_key}: {golden_path}\n"
            f"Expected:\n{_format_schema(expected_table.schema)}\n"
            f"Actual:\n{_format_schema(actual_table.schema)}"
        )
        raise AssertionError(message)

    actual_rows = actual_table.to_pylist()
    expected_rows = expected_table.to_pylist()
    if actual_rows != expected_rows:
        diff = _format_diff(_format_rows(expected_rows), _format_rows(actual_rows))
        message = f"Table output differs from golden: {golden_path}\nTable: {table_key}\n{diff}"
        raise AssertionError(message)
    return actual_table


def _ipc_bytes_for_table(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _table_from_ipc(path: Path) -> pa.Table:
    with path.open("rb") as handle:
        reader = pa.ipc.open_stream(handle)
        return reader_to_table(reader)


def _format_rows(rows: list[dict[str, object]]) -> str:
    return json.dumps(rows, indent=2, default=str).strip() + "\n"


def _format_schema(schema: pa.Schema) -> str:
    metadata = schema.metadata or {}
    decoded_metadata = {
        key.decode("utf-8"): value.decode("utf-8") for key, value in metadata.items()
    }
    base = schema.to_string(show_metadata=False) if hasattr(schema, "to_string") else str(schema)
    if not decoded_metadata:
        return base
    payload = json.dumps(decoded_metadata, indent=2, sort_keys=True)
    return f"{base}\nmetadata: {payload}"


def _format_diff(expected: str, actual: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(diff)


def _update_enabled() -> bool:
    return os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true"}


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _is_safe_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(name))


__all__ = ["assert_table_matches_golden", "dump_table"]
