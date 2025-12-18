"""V1 search HTTP routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.metrics import QueryMetrics
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.http.route_utils import schedule_query_metrics
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchQueryResponse)
async def search(
    payload: SearchQueryRequest,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SearchQueryResponse:
    """Search code metadata for the current serving snapshot.

    Parameters
    ----------
    payload
        Search request payload.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SearchQueryResponse
        Search results.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        response = await run_in_threadpool(ops.search, payload)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            QueryMetrics(
                endpoint="/search",
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                view_id=None,
                query=payload.query,
                row_count=0,
                truncated=False,
            ),
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(
        background,
        QueryMetrics(
            endpoint="/search",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=payload.query,
            row_count=len(response.results),
            truncated=response.truncated,
            engine=response.engine,
            query_hash=response.query_hash,
        ),
    )
    return response


__all__ = ["router"]
