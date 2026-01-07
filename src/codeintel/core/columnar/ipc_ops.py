"""Arrow IPC streaming helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import pyarrow as pa

from codeintel.core.columnar.schema_metadata import merge_metadata


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


def _apply_batch_metadata(
    batch: pa.RecordBatch,
    metadata: Mapping[str, object] | None,
) -> pa.RecordBatch:
    if not metadata:
        return batch
    replace = getattr(batch, "replace_schema_metadata", None)
    if not callable(replace):
        return batch
    merged = merge_metadata(batch.schema.metadata, metadata, overwrite=False)
    if merged == batch.schema.metadata:
        return batch
    return replace(merged)


def iter_ipc_stream(
    reader: pa.RecordBatchReader,
    *,
    metadata: Mapping[str, object] | None = None,
    batch_metadata: Mapping[str, object] | None = None,
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
    batch_metadata
        Optional per-record-batch metadata to inject.
    options
        Optional IPC write options.
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
        merged = merge_metadata(resolved_schema.metadata, metadata, overwrite=False)
        if merged != resolved_schema.metadata:
            resolved_schema = resolved_schema.with_metadata(merged)
    with pa.ipc.new_stream(sink, resolved_schema, options=options) as writer:
        for batch in reader:
            if cancel_check is not None:
                cancel_check()
            resolved_batch = _apply_batch_metadata(batch, batch_metadata)
            writer.write_batch(resolved_batch)
            yield from sink.drain()
    yield from sink.drain()


def write_ipc_stream(
    reader: pa.RecordBatchReader,
    *,
    sink: pa.NativeFile,
    options: pa.ipc.IpcWriteOptions | None = None,
) -> None:
    """Write a RecordBatchReader to an IPC stream sink."""
    with pa.ipc.new_stream(sink, reader.schema, options=options) as writer:
        for batch in reader:
            writer.write_batch(batch)


def read_ipc_stream(
    source: pa.NativeFile,
    *,
    options: pa.ipc.IpcReadOptions | None = None,
) -> pa.RecordBatchReader:
    """Open an IPC stream reader for the given source.

    Returns
    -------
    pyarrow.RecordBatchReader
        IPC stream reader for the source.
    """
    if options is None:
        return pa.ipc.open_stream(source)
    return pa.ipc.open_stream(source, options=options)


__all__ = [
    "ArrowIpcStreamError",
    "iter_ipc_stream",
    "read_ipc_stream",
    "write_ipc_stream",
]
