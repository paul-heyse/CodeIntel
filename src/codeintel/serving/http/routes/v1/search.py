"""V1 search HTTP routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchQueryResponse)
async def search(
    payload: SearchQueryRequest,
    kernel: Kernel,
    background: BackgroundTasks,
    request: Request,
) -> SearchQueryResponse:
    """Search code metadata for the current serving snapshot.

    Parameters
    ----------
    payload
        Search request payload.
    kernel
        Semantic query kernel.
    background
        Background task queue.
    request
        Current HTTP request.

    Returns
    -------
    SearchQueryResponse
        Search results.

    Raises
    ------
    ServingError
        When the request payload is invalid.
    TypeError
        When FastAPI fails to inject a SearchQueryRequest.
    RuntimeError
        Internal error if response is unexpectedly None (should never happen).
    """
    if not isinstance(payload, SearchQueryRequest):
        msg = "FastAPI did not provide a SearchQueryRequest model"
        raise TypeError(msg)
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    response: SearchQueryResponse | None = None
    try:
        response = await run_in_threadpool(kernel.search, payload)
    except ValueError as exc:
        raise ServingError(
            problem_type=ProblemType.INVALID_QUERY,
            title="Invalid Query",
            status=400,
            detail=str(exc),
        ) from exc
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        background.add_task(
            log_query_metrics,
            QueryMetrics(
                endpoint="/search",
                view_id=None,
                query=payload.query,
                row_count=len(response.results) if response else 0,
                truncated=response.truncated if response else False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                engine=response.engine if response else None,
            ),
        )
    # Unreachable if exception was raised; type narrowing for checker
    if response is None:  # pragma: no cover
        msg = "Unexpected state: response not set"
        raise RuntimeError(msg)
    return response


__all__ = ["router"]
