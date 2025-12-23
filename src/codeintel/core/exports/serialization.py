"""Export-focused serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

from codeintel.core.data_models.ids import normalize_decimal_id


def _format_datetime(value: datetime) -> str:
    normalized = value
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=UTC)
    else:
        normalized = normalized.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


def coerce_export_value(value: object) -> object:
    """Coerce values into JSON-compatible types for exports.

    Returns
    -------
    object
        JSON-compatible representation of the value.
    """
    if isinstance(value, datetime):
        result: object = _format_datetime(value)
    elif isinstance(value, date):
        result = value.isoformat()
    elif isinstance(value, (str, bool, int, float)) or value is None:
        result = value
    elif isinstance(value, Decimal):
        normalized = normalize_decimal_id(value)
        result = normalized if normalized is not None else str(value)
    elif isinstance(value, bytes):
        result = str(value)
    elif isinstance(value, Mapping):
        result = {
            str(key): coerce_export_value(item)
            for key, item in value.items()
        }
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = [coerce_export_value(item) for item in value]
    else:
        result = str(value)
    return result


def coerce_export_row(row: Mapping[str, object]) -> dict[str, object]:
    """Coerce a mapping into an export-ready JSON row.

    Returns
    -------
    dict[str, object]
        Row with JSON-compatible values.
    """
    return {str(key): coerce_export_value(value) for key, value in row.items()}


__all__ = ["coerce_export_row", "coerce_export_value"]
