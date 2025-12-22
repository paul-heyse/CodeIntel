"""Shared NDJSON encoding utilities for serving exports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

try:
    import msgspec

    def _enc_hook(obj: object) -> object:
        return str(obj)

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder(enc_hook=_enc_hook)
except ImportError:
    _MSG_ENCODER = None


def encode_ndjson_line(row: Mapping[str, object]) -> bytes:
    """Encode a single row as a UTF-8 NDJSON line."""
    if _MSG_ENCODER is not None:
        return _MSG_ENCODER.encode(row) + b"\n"
    payload = json.dumps(
        row,
        default=str,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return payload.encode("utf-8") + b"\n"


def iter_ndjson_bytes(rows: Iterable[Mapping[str, object]]) -> Iterator[bytes]:
    """Yield rows as UTF-8 NDJSON byte lines."""
    for row in rows:
        yield encode_ndjson_line(row)


__all__ = ["encode_ndjson_line", "iter_ndjson_bytes"]
