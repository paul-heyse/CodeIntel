"""Shared normalization helpers for docs view rows."""

from __future__ import annotations

import json
from typing import Any

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


def normalize_entrypoints_payload(value: object) -> list[dict[str, str | list[str]]]:
    """
    Normalize entrypoints payload into list-of-dicts.

    Parameters
    ----------
    value:
        Raw entrypoints value (stringified JSON, list, dict, or None).

    Returns
    -------
    list[dict[str, str | list[str]]]
        Normalized list of string-keyed dicts with string or list-of-string values.
    """
    coerced = _coerce_json(value)
    if coerced is None:
        return []
    if isinstance(coerced, dict):
        return [_coerce_entrypoint_dict(coerced)]
    if isinstance(coerced, list):
        return [_coerce_entrypoint_dict(item) for item in coerced if isinstance(item, dict)]
    return []


def _coerce_entrypoint_dict(data: dict[Any, Any]) -> dict[str, str | list[str]]:
    normalized: dict[str, str | list[str]] = {}
    for key, raw_value in data.items():
        key_str = str(key)
        if isinstance(raw_value, list):
            normalized[key_str] = [str(v) for v in raw_value]
        else:
            normalized[key_str] = str(raw_value)
    return normalized


def normalize_entrypoints_row(row: RowDict | None, *, key: str = "entrypoints_json") -> None:
    """Ensure entrypoints_json is a normalized list-of-dicts for a single row."""
    if row is None:
        return
    row[key] = normalize_entrypoints_payload(row.get(key))


def normalize_entrypoints_rows(rows: list[RowDict]) -> None:
    """Normalize entrypoints_json for a list of rows."""
    for row in rows:
        normalize_entrypoints_row(row)
