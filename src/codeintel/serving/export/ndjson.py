"""Shared JSONL encoding utilities for serving exports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from pyarrow import RecordBatch, RecordBatchReader


try:
    import orjson

    _ORJSON = orjson
except ImportError:
    _ORJSON = None

try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder()
except ImportError:
    _MSG_ENCODER = None

from codeintel.core.exports.serialization import coerce_export_row
from codeintel.storage.query_results import records_from_arrow_batch


def encode_ndjson_line(row: Mapping[str, object]) -> bytes:
    """Encode a single row as a UTF-8 JSONL line.

    Returns
    -------
    bytes
        Serialized JSONL line with a trailing newline.
    """
    payload_row = coerce_export_row(row)
    if _ORJSON is not None:
        return _ORJSON.dumps(payload_row, option=_ORJSON.OPT_APPEND_NEWLINE)
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


def iter_ndjson_bytes_from_batches(
    batches: Iterable[RecordBatch],
    *,
    cancel_check: Callable[[], None] | None = None,
    batch_hook: Callable[[RecordBatch], None] | None = None,
) -> Iterator[bytes]:
    """Yield Arrow record batches as UTF-8 JSONL byte chunks.

    Parameters
    ----------
    batches
        Record batch iterable to serialize as newline-delimited JSON.
    cancel_check
        Optional cancellation hook invoked between batches.
    batch_hook
        Optional callback invoked for each record batch.

    Yields
    ------
    bytes
        UTF-8 JSONL chunks for each record batch.

    Notes
    -----
    This function uses shared row coercion to ensure consistent export encoding.
    """
    for batch in batches:
        if cancel_check is not None:
            cancel_check()
        if batch_hook is not None:
            batch_hook(batch)
        payload = _batch_to_ndjson_bytes(batch, columns=batch.schema.names)
        if payload:
            yield payload


def iter_ndjson_bytes_from_reader(
    reader: RecordBatchReader,
    *,
    cancel_check: Callable[[], None] | None = None,
    batch_hook: Callable[[RecordBatch], None] | None = None,
) -> Iterator[bytes]:
    """Yield Arrow record batches from a reader as UTF-8 JSONL byte chunks.

    Parameters
    ----------
    reader
        Record batch reader to serialize as newline-delimited JSON.
    cancel_check
        Optional cancellation hook invoked between batches.
    batch_hook
        Optional callback invoked for each record batch.

    Yields
    ------
    bytes
        UTF-8 JSONL chunks for each record batch.
    """
    yield from iter_ndjson_bytes_from_batches(
        reader,
        cancel_check=cancel_check,
        batch_hook=batch_hook,
    )


def _batch_to_ndjson_bytes(batch: RecordBatch, *, columns: list[str]) -> bytes:
    rows = records_from_arrow_batch(batch, columns=columns)
    if not rows:
        return b""
    return b"".join(encode_ndjson_line(row) for row in rows)


__all__ = [
    "encode_ndjson_line",
    "iter_ndjson_bytes",
    "iter_ndjson_bytes_from_batches",
    "iter_ndjson_bytes_from_reader",
]
