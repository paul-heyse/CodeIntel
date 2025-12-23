"""Shared JSONL encoding utilities for serving exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from codeintel.core.exports.serialization import coerce_export_row

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder()
except ImportError:
    _MSG_ENCODER = None


def encode_ndjson_line(row: Mapping[str, object]) -> bytes:
    """Encode a single row as a UTF-8 JSONL line.

    Returns
    -------
    bytes
        Serialized JSONL line with a trailing newline.
    """
    payload_row = coerce_export_row(row)
    if _MSG_ENCODER is not None:
        return _MSG_ENCODER.encode(payload_row) + b"\n"
    payload = json.dumps(
        payload_row,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return payload.encode("utf-8") + b"\n"


def iter_ndjson_bytes(rows: Iterable[Mapping[str, object]]) -> Iterator[bytes]:
    """Yield rows as UTF-8 JSONL byte lines.

    Yields
    ------
    bytes
        Serialized JSONL line with a trailing newline.
    """
    for row in rows:
        yield encode_ndjson_line(row)


__all__ = ["encode_ndjson_line", "iter_ndjson_bytes"]
