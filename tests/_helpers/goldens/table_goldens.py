"""Golden helpers for table dumps."""

from __future__ import annotations

import difflib
import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def dump_table(
    gateway: StorageGateway,
    table_key: str,
) -> list[dict[str, object]]:
    """Load and normalize a table into stable, JSON-friendly records.

    Parameters
    ----------
    gateway
        Storage gateway to query.
    table_key
        Schema-qualified table key.

    Returns
    -------
    list[dict[str, object]]
        Sorted, normalized row dictionaries.

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
    rows = relation.fetchall()
    columns = list(relation.columns)
    records = [
        {columns[idx]: _normalize_value(value) for idx, value in enumerate(row)} for row in rows
    ]
    return sorted(records, key=_sort_key)


def assert_table_matches_golden(
    gateway: StorageGateway,
    table_key: str,
    *,
    golden_path: Path,
    update_mode: bool | None = None,
) -> list[dict[str, object]]:
    """Assert a table dump matches a golden JSON file.

    Parameters
    ----------
    gateway
        Storage gateway to query.
    table_key
        Schema-qualified table key.
    golden_path
        Expected golden JSON file path.
    update_mode
        When True, overwrite golden files with current output.

    Returns
    -------
    list[dict[str, object]]
        Normalized rows used for the comparison.

    Raises
    ------
    AssertionError
        If the golden file is missing or differs from current output.
    """
    records = dump_table(gateway, table_key)
    actual = _format_json(records)
    should_update = update_mode if update_mode is not None else _update_enabled()

    if should_update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual, encoding="utf-8")
        return records

    if not golden_path.exists():
        message = (
            f"Golden file not found: {golden_path}\n"
            "Run with UPDATE_GOLDEN=1 to create it.\n"
            f"Actual output:\n{actual}"
        )
        raise AssertionError(message)

    expected = _format_json(json.loads(golden_path.read_text(encoding="utf-8")))
    if actual != expected:
        diff = _format_diff(expected, actual)
        message = f"Table output differs from golden: {golden_path}\nTable: {table_key}\n{diff}"
        raise AssertionError(message)
    return records


def _format_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True).strip() + "\n"


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


def _normalize_value(value: object) -> object:
    if isinstance(value, datetime):
        normalized: object = value.isoformat()
    elif isinstance(value, date):
        normalized = value.isoformat()
    elif isinstance(value, (Decimal, Path, UUID)):
        normalized = str(value)
    elif isinstance(value, bytes):
        normalized = value.hex()
    elif isinstance(value, dict):
        normalized = {key: _normalize_value(val) for key, val in value.items()}
    elif isinstance(value, (tuple, list)):
        normalized = [_normalize_value(item) for item in value]
    else:
        normalized = value
    return normalized


def _sort_key(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True, default=str)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _is_safe_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(name))


__all__ = ["assert_table_matches_golden", "dump_table"]
