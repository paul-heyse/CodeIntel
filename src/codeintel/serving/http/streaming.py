"""Streaming response utilities for large resultsets.

This module provides utilities for streaming query results as newline-delimited
JSON (NDJSON) to support efficient export of large datasets.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from starlette.responses import StreamingResponse

from codeintel.serving.export.formats import mime_type_for_export_format

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

try:
    import msgspec

    def _enc_hook(obj: object) -> object:
        return str(obj)

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder(enc_hook=_enc_hook)
except ImportError:
    _MSG_ENCODER = None


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
    for row in rows:
        if _MSG_ENCODER is not None:
            yield _MSG_ENCODER.encode(row) + b"\n"
        else:
            yield json.dumps(
                row,
                default=str,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8") + b"\n"


def ndjson_response(
    rows: Iterable[dict[str, object]],
    *,
    filename: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    """Create an NDJSON streaming response.

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
        Streaming response with NDJSON content type.
    """
    response_headers: dict[str, str] = {}
    if filename:
        response_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if headers is not None:
        response_headers.update({str(k): str(v) for k, v in headers.items()})
    return StreamingResponse(
        ndjson_stream(rows),
        media_type=mime_type_for_export_format("ndjson"),
        headers=response_headers,
    )


__all__ = ["ndjson_response", "ndjson_stream"]
