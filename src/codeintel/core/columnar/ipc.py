"""Arrow IPC schema serialization helpers."""

from __future__ import annotations

import base64
import binascii

import pyarrow as pa


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
        return buffer.to_pybytes()
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


__all__ = ["schema_from_ipc_payload", "schema_to_ipc_payload"]
