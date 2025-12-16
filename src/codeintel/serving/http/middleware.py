"""Serving-specific HTTP middleware."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request, Response

CORRELATION_ID_HEADER = "X-Correlation-ID"


def get_correlation_id(request: Request) -> str:
    """Return the correlation ID for a request.

    Parameters
    ----------
    request
        Current request.

    Returns
    -------
    str
        Correlation ID.

    Raises
    ------
    RuntimeError
        If the correlation ID is missing (middleware not installed).
    """
    raw = getattr(request.state, "correlation_id", None)
    if isinstance(raw, str) and raw:
        return raw
    msg = "Correlation ID missing; serving middleware not installed"
    raise RuntimeError(msg)


async def correlation_id_and_timing_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach correlation ID and timing headers to all responses.

    Parameters
    ----------
    request
        Current request.
    call_next
        Starlette request handler.

    Returns
    -------
    Response
        Response with correlation and timing headers applied.
    """
    start = time.perf_counter()

    incoming = request.headers.get(CORRELATION_ID_HEADER)
    correlation_id = incoming.strip() if incoming else uuid.uuid4().hex
    request.state.correlation_id = correlation_id

    response = await call_next(request)
    response.headers[CORRELATION_ID_HEADER] = correlation_id

    elapsed_s = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed_s:.6f}"
    response.headers["Server-Timing"] = f"app;dur={elapsed_s * 1000:.3f}"
    return response


__all__ = [
    "CORRELATION_ID_HEADER",
    "correlation_id_and_timing_middleware",
    "get_correlation_id",
]
