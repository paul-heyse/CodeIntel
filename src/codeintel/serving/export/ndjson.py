"""Shared JSONL encoding utilities for serving exports."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.json as pa_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pyarrow import RecordBatch, RecordBatchReader


try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder()
except ImportError:
    _MSG_ENCODER = None

from codeintel.core.exports.serialization import coerce_export_row


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


def iter_ndjson_bytes_from_reader(reader: RecordBatchReader) -> Iterator[bytes]:
    """Yield Arrow record batches as UTF-8 JSONL byte chunks.

    Parameters
    ----------
    reader
        Record batch reader to serialize as newline-delimited JSON.

    Yields
    ------
    bytes
        UTF-8 JSONL chunks for each record batch.

    Raises
    ------
    TypeError
        If ``pyarrow.json.write_json`` is unavailable in the runtime.
    """
    write_json = getattr(pa_json, "write_json", None)
    if not callable(write_json):
        msg = "pyarrow.json.write_json is unavailable"
        raise TypeError(msg)
    write_json_fn = cast("Callable[[pa.Table, pa.BufferOutputStream], None]", write_json)
    for batch in reader:
        yield from _batch_to_ndjson_bytes(batch, schema=reader.schema, write_json=write_json_fn)


def _batch_to_ndjson_bytes(
    batch: RecordBatch,
    *,
    schema: pa.Schema,
    write_json: Callable[[pa.Table, pa.BufferOutputStream], None],
) -> Iterator[bytes]:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_batches([batch], schema=schema)
    write_json(table, sink)
    payload = sink.getvalue().to_pybytes()
    if payload:
        yield payload


__all__ = ["encode_ndjson_line", "iter_ndjson_bytes", "iter_ndjson_bytes_from_reader"]
