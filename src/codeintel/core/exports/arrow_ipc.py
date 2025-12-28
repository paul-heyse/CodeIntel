"""Arrow IPC stream helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping

import pyarrow as pa

ARROW_IPC_STREAM_MIME = "application/vnd.apache.arrow.stream"


class ArrowIpcStreamError(ValueError):
    """Raised when Arrow IPC streaming fails."""


class _ChunkedIpcSink:
    """File-like sink that buffers IPC writes into discrete chunks."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._closed = False
        self._size = 0

    def write(self, data: bytes | memoryview) -> int:
        if self._closed:
            msg = "Arrow IPC sink is closed"
            raise ArrowIpcStreamError(msg)
        if not data:
            return 0
        chunk = bytes(data)
        self._chunks.append(chunk)
        self._size += len(chunk)
        return len(chunk)

    def flush(self) -> None:
        if self._closed:
            return

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self._size

    def drain(self) -> Iterator[bytes]:
        chunks = self._chunks
        self._chunks = []
        yield from chunks


def default_ipc_write_options() -> pa.ipc.IpcWriteOptions:
    """Return default Arrow IPC write options.

    Returns
    -------
    pyarrow.ipc.IpcWriteOptions
        Default IPC write options with compression and metadata version.
    """
    return pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V5,
        compression="zstd",
        use_threads=True,
        unify_dictionaries=True,
    )


def _encode_metadata_value(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): _encode_metadata_value(value) for key, value in metadata.items()
    }


def _merge_schema_metadata(schema: pa.Schema, metadata: Mapping[str, object]) -> pa.Schema:
    existing = schema.metadata or {}
    merged = dict(existing)
    merged.update(_encode_metadata(metadata))
    return schema.with_metadata(merged)


def apply_ipc_metadata(schema: pa.Schema, metadata: Mapping[str, object] | None) -> pa.Schema:
    """Return schema with serialized metadata applied.

    Parameters
    ----------
    schema
        Base Arrow schema.
    metadata
        Optional metadata to inject into the schema.

    Returns
    -------
    pyarrow.Schema
        Schema with merged metadata.
    """
    if not metadata:
        return schema
    return _merge_schema_metadata(schema, metadata)


def iter_ipc_stream(
    reader: pa.RecordBatchReader,
    *,
    metadata: Mapping[str, object] | None = None,
    options: pa.ipc.IpcWriteOptions | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Iterator[bytes]:
    """Yield Arrow IPC stream bytes for a RecordBatchReader.

    Parameters
    ----------
    reader
        RecordBatchReader to serialize.
    metadata
        Optional schema metadata to inject into the IPC stream.
    options
        Optional IPC write options. Uses default options when omitted.
    cancel_check
        Optional cancellation hook invoked between record batches.

    Yields
    ------
    bytes
        IPC stream bytes chunk-by-chunk.
    """
    sink = _ChunkedIpcSink()
    resolved_schema = reader.schema
    if metadata:
        resolved_schema = _merge_schema_metadata(resolved_schema, metadata)
    write_options = options or default_ipc_write_options()
    with pa.ipc.new_stream(sink, resolved_schema, options=write_options) as writer:
        for batch in reader:
            if cancel_check is not None:
                cancel_check()
            writer.write_batch(batch)
            yield from sink.drain()
    yield from sink.drain()


__all__ = [
    "ARROW_IPC_STREAM_MIME",
    "ArrowIpcStreamError",
    "apply_ipc_metadata",
    "default_ipc_write_options",
    "iter_ipc_stream",
]
