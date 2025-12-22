"""Shared NDJSON encoding utilities for serving exports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _format_datetime(value: datetime) -> str:
    normalized = value
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=UTC)
    else:
        normalized = normalized.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


def _coerce_ndjson_value(value: object) -> object:
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, bytes):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _coerce_ndjson_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_ndjson_value(item) for item in value]
    return str(value)


def _coerce_ndjson_row(row: Mapping[str, object]) -> dict[str, object]:
    return {key: _coerce_ndjson_value(value) for key, value in row.items()}


try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder(enc_hook=_coerce_ndjson_value)
except ImportError:
    _MSG_ENCODER = None


def encode_ndjson_line(row: Mapping[str, object]) -> bytes:
    """Encode a single row as a UTF-8 NDJSON line.

    Returns
    -------
    bytes
        Serialized NDJSON line with a trailing newline.
    """
    payload_row = _coerce_ndjson_row(row)
    if _MSG_ENCODER is not None:
        return _MSG_ENCODER.encode(payload_row) + b"\n"
    payload = json.dumps(
        payload_row,
        default=str,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return payload.encode("utf-8") + b"\n"


def iter_ndjson_bytes(rows: Iterable[Mapping[str, object]]) -> Iterator[bytes]:
    """Yield rows as UTF-8 NDJSON byte lines.

    Yields
    ------
    bytes
        Serialized NDJSON line with a trailing newline.
    """
    for row in rows:
        yield encode_ndjson_line(row)


__all__ = ["encode_ndjson_line", "iter_ndjson_bytes"]
