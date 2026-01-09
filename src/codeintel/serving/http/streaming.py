"""Streaming response utilities for large resultsets.

This module provides utilities for streaming query results as newline-delimited
JSON (JSONL) or Arrow IPC streams to support efficient export of large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from codeintel.core.columnar.arrowdsl import (
    ExecutionContext,
    ExecutionPlan,
    PipelineRunOptions,
    run_pipeline,
)
from codeintel.core.columnar.conversion import (
    record_batch_reader_from_iterable,
    table_from_batches,
)
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.columnar.streaming import ScanTelemetry
from codeintel.core.exports import ARROW_IPC_STREAM_MIME, iter_ipc_stream
from codeintel.serving.export.formats import mime_type_for_export_format
from codeintel.serving.export.ndjson import (
    NdjsonBatchOptions,
    iter_ndjson_bytes,
    iter_ndjson_bytes_from_batches,
    iter_ndjson_bytes_from_reader,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from pyarrow import RecordBatch

    from codeintel.core.columnar.finalize_ops import FinalizeResult, FinalizeSpec


def ndjson_stream(rows: Iterable[dict[str, object]]) -> Iterator[bytes]:
    """Yield rows as newline-delimited JSON bytes.

    Parameters
    ----------
    rows
        Iterable of row dictionaries to stream.

    Yields
    ------
    bytes
        JSON-encoded row followed by newline.
    """
    yield from iter_ndjson_bytes(rows)


def ndjson_response(
    rows: Iterable[dict[str, object]],
    *,
    filename: str | None = None,
    headers: Mapping[str, str] | None = None,
    background: BackgroundTask | None = None,
) -> StreamingResponse:
    """Create a JSONL streaming response.

    Parameters
    ----------
    rows
        Iterable of row dictionaries to stream.
    filename
        Optional filename for Content-Disposition header.
    headers
        Optional extra response headers.
    background
        Optional background task run after the response completes.

    Returns
    -------
    StreamingResponse
        Streaming response with JSONL content type.
    """
    response_headers: dict[str, str] = {}
    if filename:
        response_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if headers is not None:
        response_headers.update({str(k): str(v) for k, v in headers.items()})
    return StreamingResponse(
        ndjson_stream(rows),
        media_type=mime_type_for_export_format("jsonl"),
        headers=response_headers,
        background=background,
    )


def ndjson_response_from_batches(
    batches: Iterable[RecordBatch],
    *,
    options: NdjsonBatchResponseOptions | None = None,
) -> StreamingResponse:
    """Create a JSONL streaming response from Arrow record batches.

    Parameters
    ----------
    batches
        Record batch iterable to serialize as JSONL.
    options
        Optional response options for filename, headers, cancellation, and hooks.

    Returns
    -------
    StreamingResponse
        Streaming response with JSONL content type.
    """
    resolved = options or NdjsonBatchResponseOptions()
    response_headers = _response_headers(
        filename=resolved.filename,
        headers=resolved.headers,
    )
    payload = iter_ndjson_bytes_from_batches(
        batches,
        options=NdjsonBatchOptions(
            cancel_check=resolved.cancel_check,
            batch_hook=resolved.batch_hook,
            finalize_spec=resolved.finalize_spec,
            finalize_hook=resolved.finalize_hook,
            execution_ctx=resolved.execution_context,
            manifest_dir=resolved.manifest_dir,
            manifest_options=resolved.manifest_options,
            scan_telemetry=resolved.scan_telemetry,
        ),
    )
    return StreamingResponse(
        payload,
        media_type=mime_type_for_export_format("jsonl"),
        headers=response_headers,
        background=resolved.background,
    )


def _response_headers(
    *,
    filename: str | None,
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    response_headers: dict[str, str] = {}
    if filename:
        response_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if headers is not None:
        response_headers.update({str(k): str(v) for k, v in headers.items()})
    return response_headers


def _ipc_stream_supported() -> bool:
    stream_fn = getattr(pa.ipc, "new_stream", None)
    return callable(stream_fn)


def _fallback_filename(filename: str | None) -> str | None:
    if filename is None:
        return None
    for suffix in (".arrow", ".ipc", ".feather"):
        if filename.endswith(suffix):
            return f"{filename[: -len(suffix)]}.jsonl"
    return filename


@dataclass(frozen=True, slots=True)
class ArrowIpcResponseOptions:
    """Options for Arrow IPC streaming responses."""

    filename: str | None = None
    headers: Mapping[str, str] | None = None
    metadata: Mapping[str, object] | None = None
    batch_metadata: Mapping[str, object] | None = None
    options: pa.ipc.IpcWriteOptions | None = None
    cancel_check: Callable[[], None] | None = None
    background: BackgroundTask | None = None
    finalize_spec: FinalizeSpec | None = None
    finalize_hook: Callable[[FinalizeResult], None] | None = None
    execution_context: ExecutionContext | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None
    scan_telemetry: ScanTelemetry | None = None


@dataclass(frozen=True, slots=True)
class NdjsonBatchResponseOptions:
    """Options for NDJSON batch streaming responses."""

    filename: str | None = None
    headers: Mapping[str, str] | None = None
    background: BackgroundTask | None = None
    cancel_check: Callable[[], None] | None = None
    batch_hook: Callable[[RecordBatch], None] | None = None
    finalize_spec: FinalizeSpec | None = None
    finalize_hook: Callable[[FinalizeResult], None] | None = None
    execution_context: ExecutionContext | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None
    scan_telemetry: ScanTelemetry | None = None


def arrow_ipc_response(
    source: pa.RecordBatchReader | Iterable[bytes],
    *,
    options: ArrowIpcResponseOptions | None = None,
) -> StreamingResponse:
    """Create an Arrow IPC streaming response.

    Parameters
    ----------
    source
        RecordBatchReader or pre-encoded IPC byte chunks to stream.
    options
        Optional Arrow IPC response options.

    Returns
    -------
    StreamingResponse
        Streaming response with Arrow IPC stream content type.
    """
    resolved = options or ArrowIpcResponseOptions()
    if isinstance(source, pa.RecordBatchReader) and not _ipc_stream_supported():
        response_headers = _response_headers(
            filename=_fallback_filename(resolved.filename),
            headers=resolved.headers,
        )
        return StreamingResponse(
            iter_ndjson_bytes_from_reader(
                source,
                options=NdjsonBatchOptions(
                    cancel_check=resolved.cancel_check,
                    finalize_spec=resolved.finalize_spec,
                    finalize_hook=resolved.finalize_hook,
                    execution_ctx=resolved.execution_context,
                ),
            ),
            media_type=mime_type_for_export_format("jsonl"),
            headers=response_headers,
            background=resolved.background,
        )
    response_headers = _response_headers(
        filename=resolved.filename,
        headers=resolved.headers,
    )
    if isinstance(source, pa.RecordBatchReader):
        reader = source
        if resolved.finalize_spec is not None:
            reader = _finalized_reader(
                reader,
                request=_FinalizeReaderRequest(
                    finalize_spec=resolved.finalize_spec,
                    finalize_hook=resolved.finalize_hook,
                    cancel_check=resolved.cancel_check,
                    execution_ctx=resolved.execution_context,
                    manifest_dir=resolved.manifest_dir,
                    manifest_options=resolved.manifest_options,
                    scan_telemetry=resolved.scan_telemetry,
                ),
            )
        payload = iter_ipc_stream(
            reader,
            metadata=resolved.metadata,
            batch_metadata=resolved.batch_metadata,
            options=resolved.options,
            cancel_check=resolved.cancel_check,
        )
    else:
        payload = source
    return StreamingResponse(
        payload,
        media_type=ARROW_IPC_STREAM_MIME,
        headers=response_headers,
        background=resolved.background,
    )


@dataclass(frozen=True, slots=True)
class _FinalizeReaderRequest:
    finalize_spec: FinalizeSpec
    finalize_hook: Callable[[FinalizeResult], None] | None
    cancel_check: Callable[[], None] | None
    execution_ctx: ExecutionContext | None
    manifest_dir: Path | None
    manifest_options: RunManifestOptions | None
    scan_telemetry: ScanTelemetry | None


def _finalized_reader(
    reader: pa.RecordBatchReader,
    *,
    request: _FinalizeReaderRequest,
) -> pa.RecordBatchReader:
    resolved_ctx = request.execution_ctx or ExecutionContext()

    def _iter_batches() -> Iterator[RecordBatch]:
        for batch in reader:
            if request.cancel_check is not None:
                request.cancel_check()
            if batch.num_rows == 0:
                continue
            table = table_from_batches([batch], schema=batch.schema)
            result = run_pipeline(
                plan=ExecutionPlan.from_table(table),
                finalize=request.finalize_spec,
                options=PipelineRunOptions(
                    ctx=resolved_ctx,
                    manifest_dir=request.manifest_dir,
                    manifest_options=request.manifest_options,
                    scan_telemetry=request.scan_telemetry,
                ),
            )
            if request.finalize_hook is not None:
                request.finalize_hook(result)
            yield from result.good.to_batches(max_chunksize=batch.num_rows)

    finalized = record_batch_reader_from_iterable(_iter_batches(), empty_policy="none")
    if finalized is None:
        return empty_reader_from_schema(reader.schema)
    return finalized


__all__ = [
    "ArrowIpcResponseOptions",
    "NdjsonBatchResponseOptions",
    "arrow_ipc_response",
    "ndjson_response",
    "ndjson_response_from_batches",
    "ndjson_stream",
]
