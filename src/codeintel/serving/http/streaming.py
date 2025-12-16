"""Streaming response utilities for large resultsets.

This module provides utilities for streaming query results as newline-delimited
JSON (NDJSON) to support efficient export of large datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING

from starlette.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        yield json.dumps(row, default=str).encode("utf-8") + b"\n"


def ndjson_response(
    rows: Iterable[dict[str, object]],
    *,
    filename: str | None = None,
) -> StreamingResponse:
    """Create an NDJSON streaming response.

    Parameters
    ----------
    rows
        Iterable of row dictionaries to stream.
    filename
        Optional filename for Content-Disposition header.

    Returns
    -------
    StreamingResponse
        Streaming response with NDJSON content type.
    """
    headers: dict[str, str] = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return StreamingResponse(
        ndjson_stream(rows),
        media_type="application/x-ndjson",
        headers=headers,
    )


__all__ = ["ndjson_response", "ndjson_stream"]
