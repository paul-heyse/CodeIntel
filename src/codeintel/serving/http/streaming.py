"""Streaming response utilities for large resultsets.

This module provides utilities for streaming query results as newline-delimited
JSON (JSONL) or Arrow IPC streams to support efficient export of large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from starlette.responses import StreamingResponse

from codeintel.core.exports import ARROW_IPC_STREAM_MIME, iter_ipc_stream
from codeintel.serving.export.formats import mime_type_for_export_format
from codeintel.serving.export.ndjson import iter_ndjson_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping


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
    )


@dataclass(frozen=True, slots=True)
class ArrowIpcResponseOptions:
    filename: str | None = None
    headers: Mapping[str, str] | None = None
    metadata: Mapping[str, object] | None = None
    batch_metadata: Mapping[str, object] | None = None
    options: pa.ipc.IpcWriteOptions | None = None
    cancel_check: Callable[[], None] | None = None


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
    response_headers: dict[str, str] = {}
    if resolved.filename:
        response_headers["Content-Disposition"] = f'attachment; filename="{resolved.filename}"'
    if resolved.headers is not None:
        response_headers.update({str(k): str(v) for k, v in resolved.headers.items()})
    if isinstance(source, pa.RecordBatchReader):
        payload = iter_ipc_stream(
            source,
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
    )


__all__ = [
    "ArrowIpcResponseOptions",
    "arrow_ipc_response",
    "ndjson_response",
    "ndjson_stream",
]
