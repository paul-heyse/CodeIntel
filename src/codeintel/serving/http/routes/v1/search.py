"""V1 search HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.route_utils import (
    ThreadpoolMetricsContext,
    run_in_threadpool_with_metrics,
)
from codeintel.serving.metrics import QueryMetrics
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse

router = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(require_api_key, scope="request")],
)


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

    def _success(
        response: SearchQueryResponse, duration_ms: float, correlation_id: str
    ) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/search",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=payload.query,
            row_count=len(response.results),
            truncated=response.truncated,
            engine=response.engine,
            query_hash=response.query_hash,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/search",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=payload.query,
            row_count=0,
            truncated=False,
        )

    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success,
        error_metrics=_error,
    )
    return await run_in_threadpool_with_metrics(context, ops.search, payload)


__all__ = ["router"]
