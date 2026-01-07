"""Arrow IPC schema serialization helpers."""

from __future__ import annotations

import base64
import binascii
from typing import BinaryIO

import pyarrow as pa

from codeintel.core.columnar import ipc_ops as _ipc_ops


def schema_from_ipc_payload(payload: str) -> pa.Schema | None:
    """Decode a base64 Arrow IPC schema payload.

    Parameters
    ----------
    payload
        Base64-encoded Arrow IPC schema bytes.

    Returns
    -------
    pyarrow.Schema | None
        Decoded schema when valid, otherwise None.
    """
    try:
        raw = base64.b64decode(payload)
    except (ValueError, binascii.Error):
        return None
    try:
        buffer = pa.py_buffer(raw)
        return pa.ipc.read_schema(pa.BufferReader(buffer))
    except (OSError, pa.ArrowInvalid, ValueError):
        return None


def schema_to_ipc_payload(schema: pa.Schema) -> str:
    """Encode a schema into a base64 Arrow IPC payload.

    Parameters
    ----------
    schema
        Arrow schema to serialize.

    Returns
    -------
    str
        Base64-encoded Arrow IPC schema bytes.
    """
    return base64.b64encode(_serialize_schema_ipc(schema)).decode("ascii")


def _serialize_schema_ipc(schema: pa.Schema) -> bytes:
    serialize_schema = getattr(pa.ipc, "serialize_schema", None)
    if callable(serialize_schema):
        buffer = serialize_schema(schema)
        to_pybytes = getattr(buffer, "to_pybytes", None)
        if callable(to_pybytes):
            result = to_pybytes()
            if isinstance(result, (bytes, bytearray)):
                return bytes(result)
        if isinstance(buffer, (bytes, bytearray)):
            return bytes(buffer)
        msg = "Arrow IPC schema serialization returned unsupported buffer type"
        raise TypeError(msg)
    write_schema = getattr(pa.ipc, "write_schema", None)
    if callable(write_schema):
        sink = pa.BufferOutputStream()
        write_schema(schema, sink)
        return sink.getvalue().to_pybytes()
    new_stream = getattr(pa.ipc, "new_stream", None)
    if callable(new_stream):
        sink = pa.BufferOutputStream()
        writer = new_stream(sink, schema)
        close = getattr(writer, "close", None)
        if callable(close):
            close()
        return sink.getvalue().to_pybytes()
    msg = "Arrow IPC schema serialization is unavailable"
    raise TypeError(msg)


def write_ipc_stream(
    reader: pa.RecordBatchReader,
    writer: BinaryIO,
    *,
    options: pa.ipc.IpcWriteOptions | None = None,
) -> None:
    """Write an Arrow IPC stream to a binary writer.

    Parameters
    ----------
    reader
        RecordBatchReader providing stream batches.
    writer
        Binary writer (e.g., sys.stdout.buffer).
    options
        Optional IPC write options to apply.
    """
    sink = pa.output_stream(writer)
    _ipc_ops.write_ipc_stream(reader, sink=sink, options=options)


__all__ = ["schema_from_ipc_payload", "schema_to_ipc_payload", "write_ipc_stream"]
