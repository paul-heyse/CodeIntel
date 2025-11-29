"""Shared normalization helpers for docs view rows."""

from __future__ import annotations

import json

from codeintel.storage.repositories.base import RowDict


def _coerce_json(value: object) -> object:
    """
    Attempt to parse JSON strings into Python objects.

    Returns
    -------
    object
        Parsed JSON value when possible, otherwise the original input.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalize_entrypoints_row(row: RowDict | None) -> None:
    """Ensure entrypoints_json is a list for a single row."""
    if row is None:
        return
    raw = row.get("entrypoints_json")
    coerced = _coerce_json(raw)
    if coerced is None:
        row["entrypoints_json"] = []
    elif isinstance(coerced, list):
        row["entrypoints_json"] = coerced
    else:
        row["entrypoints_json"] = [coerced]


def normalize_entrypoints_rows(rows: list[RowDict]) -> None:
    """Normalize entrypoints_json for a list of rows."""
    for row in rows:
        normalize_entrypoints_row(row)
