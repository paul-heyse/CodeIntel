"""Arrow IPC stream helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from inspect import signature

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
    return build_ipc_write_options(
        compression="zstd",
        use_threads=True,
        unify_dictionaries=True,
        metadata_version="V5",
    )


def build_ipc_write_options(
    *,
    compression: str | None,
    use_threads: bool | None,
    unify_dictionaries: bool | None,
    metadata_version: str | None,
) -> pa.ipc.IpcWriteOptions:
    """Build IPC write options from optional overrides.

    Parameters
    ----------
    compression
        Optional compression codec (e.g., ``"zstd"``).
    use_threads
        Whether to enable threaded IPC encoding.
    unify_dictionaries
        Whether to unify dictionary-encoded columns.
    metadata_version
        Optional metadata version override (e.g., ``"V5"``).

    Returns
    -------
    pyarrow.ipc.IpcWriteOptions
        IPC write options configured with overrides.
    """
    kwargs: dict[str, object] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if use_threads is not None:
        kwargs["use_threads"] = use_threads
    if unify_dictionaries is not None:
        kwargs["unify_dictionaries"] = unify_dictionaries
    resolved_version = _parse_metadata_version(metadata_version)
    if resolved_version is not None:
        kwargs["metadata_version"] = resolved_version
    filtered = _filter_kwargs(pa.ipc.IpcWriteOptions, kwargs)
    return pa.ipc.IpcWriteOptions(**filtered)


def build_ipc_read_options(
    *,
    use_threads: bool | None,
    max_recursion_depth: int | None,
) -> pa.ipc.IpcReadOptions | None:
    """Build IPC read options from optional overrides.

    Parameters
    ----------
    use_threads
        Whether to enable threaded IPC decoding.
    max_recursion_depth
        Optional recursion depth limit for nested data.

    Returns
    -------
    pyarrow.ipc.IpcReadOptions | None
        Read options when supported, otherwise None.
    """
    read_options = getattr(pa.ipc, "IpcReadOptions", None)
    if read_options is None or not callable(read_options):
        return None
    kwargs: dict[str, object] = {}
    if use_threads is not None:
        kwargs["use_threads"] = use_threads
    if max_recursion_depth is not None:
        kwargs["max_recursion_depth"] = max_recursion_depth
    filtered = _filter_kwargs(read_options, kwargs)
    return read_options(**filtered)


def _parse_metadata_version(value: str | None) -> pa.ipc.MetadataVersion | None:
    if not value:
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    if normalized.isdigit():
        normalized = f"V{normalized}"
    if not normalized.startswith("V"):
        normalized = f"V{normalized}"
    return getattr(pa.ipc.MetadataVersion, normalized, None)


def _filter_kwargs(
    target: Callable[..., object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    try:
        params = signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _encode_metadata_value(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes]:
    return {
        str(key).encode("utf-8"): _encode_metadata_value(value) for key, value in metadata.items()
    }


def _merge_schema_metadata(
    schema: pa.Schema,
    metadata: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> pa.Schema:
    existing = schema.metadata or {}
    merged = dict(existing)
    encoded = _encode_metadata(metadata)
    for key, value in encoded.items():
        if not overwrite and key in merged and not key.startswith(b"codeintel."):
            continue
        merged[key] = value
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
        Schema with appended metadata.
    """
    if not metadata:
        return schema
    return _merge_schema_metadata(schema, metadata)


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
            resolved_batch = _apply_batch_metadata(batch, batch_metadata)
            writer.write_batch(resolved_batch)
            yield from sink.drain()
    yield from sink.drain()


def _apply_batch_metadata(
    batch: pa.RecordBatch,
    metadata: Mapping[str, object] | None,
) -> pa.RecordBatch:
    if not metadata:
        return batch
    replace = getattr(batch, "replace_schema_metadata", None)
    if not callable(replace):
        return batch
    existing = batch.schema.metadata or {}
    merged = dict(existing)
    encoded = _encode_metadata(metadata)
    for key, value in encoded.items():
        if key in merged:
            continue
        merged[key] = value
    return replace(merged)


__all__ = [
    "ARROW_IPC_STREAM_MIME",
    "ArrowIpcStreamError",
    "apply_ipc_metadata",
    "build_ipc_read_options",
    "build_ipc_write_options",
    "default_ipc_write_options",
    "iter_ipc_stream",
]
