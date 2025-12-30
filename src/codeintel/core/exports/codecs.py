"""Export codec registry and shared serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.json as pa_json

from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyarrow import RecordBatch, RecordBatchReader


try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder()
except ImportError:
    _MSG_ENCODER = None

CodecName = str


@dataclass(frozen=True, slots=True)
class ExportCodec:
    """Codec for serializing rows and Arrow record batches."""

    name: CodecName
    encode_row: Callable[[Mapping[str, object]], bytes]
    encode_batch: Callable[[RecordBatch, pa.Schema], Iterable[bytes]]
    encode_reader: Callable[[RecordBatchReader], Iterable[bytes]]


_CODEC_REGISTRY: dict[CodecName, ExportCodec] = {}
_DEFAULT_CODEC = "ndjson"


class ExportCodecError(RuntimeError):
    """Error raised when export codec lookups fail."""


def register_export_codec(codec: ExportCodec) -> None:
    """Register an export codec by name.

    Parameters
    ----------
    codec
        Codec instance to register.
    """
    _CODEC_REGISTRY[codec.name] = codec


def get_export_codec(name: CodecName) -> ExportCodec:
    """Fetch a registered export codec.

    Parameters
    ----------
    name
        Codec name.

    Returns
    -------
    ExportCodec
        Registered codec.

    Raises
    ------
    ExportCodecError
        If the codec is not registered.
    """
    codec = _CODEC_REGISTRY.get(name)
    if codec is None:
        msg = f"Export codec not registered: {name}"
        raise ExportCodecError(msg)
    return codec


def encode_row(
    row: Mapping[str, object],
    *,
    codec: CodecName = _DEFAULT_CODEC,
) -> bytes:
    """Encode a mapping using a registered codec.

    Parameters
    ----------
    row
        Row mapping to encode.
    codec
        Codec name to use.

    Returns
    -------
    bytes
        Encoded payload for the row.
    """
    return get_export_codec(codec).encode_row(row)


def encode_batch(
    batch: RecordBatch,
    *,
    schema: pa.Schema,
    codec: CodecName = _DEFAULT_CODEC,
) -> Iterable[bytes]:
    """Encode an Arrow RecordBatch using a registered codec.

    Parameters
    ----------
    batch
        RecordBatch to encode.
    schema
        Schema to use when encoding the batch.
    codec
        Codec name to use.

    Returns
    -------
    Iterable[bytes]
        Encoded payload chunks.
    """
    return get_export_codec(codec).encode_batch(batch, schema)


def encode_reader(
    reader: RecordBatchReader,
    *,
    codec: CodecName = _DEFAULT_CODEC,
) -> Iterable[bytes]:
    """Encode a RecordBatchReader using a registered codec.

    Parameters
    ----------
    reader
        RecordBatchReader to encode.
    codec
        Codec name to use.

    Returns
    -------
    Iterable[bytes]
        Encoded payload chunks.
    """
    return get_export_codec(codec).encode_reader(reader)


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
        result = {str(key): coerce_export_value(item) for key, item in value.items()}
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

    """
    write_json = _require_write_json()
    for batch in reader:
        yield from _batch_to_ndjson_bytes(batch, schema=reader.schema, write_json=write_json)


def _require_write_json() -> Callable[[pa.Table, pa.BufferOutputStream], None]:
    write_json = getattr(pa_json, "write_json", None)
    if not callable(write_json):
        msg = "pyarrow.json.write_json is unavailable"
        raise TypeError(msg)
    return cast("Callable[[pa.Table, pa.BufferOutputStream], None]", write_json)


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


def _encode_ndjson_batch(batch: RecordBatch, schema: pa.Schema) -> Iterable[bytes]:
    write_json = _require_write_json()
    return _batch_to_ndjson_bytes(batch, schema=schema, write_json=write_json)


def _encode_ndjson_reader(reader: RecordBatchReader) -> Iterable[bytes]:
    return iter_ndjson_bytes_from_reader(reader)


register_export_codec(
    ExportCodec(
        name="ndjson",
        encode_row=encode_ndjson_line,
        encode_batch=_encode_ndjson_batch,
        encode_reader=_encode_ndjson_reader,
    )
)

__all__ = [
    "CodecName",
    "ExportCodec",
    "ExportCodecError",
    "coerce_export_row",
    "coerce_export_value",
    "encode_batch",
    "encode_ndjson_line",
    "encode_reader",
    "encode_row",
    "get_export_codec",
    "iter_ndjson_bytes",
    "iter_ndjson_bytes_from_reader",
    "register_export_codec",
]
