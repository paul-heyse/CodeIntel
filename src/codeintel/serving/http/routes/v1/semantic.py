"""V1 semantic HTTP routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.metrics import QueryMetrics
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.http.route_utils import schedule_query_metrics
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)

router = APIRouter(prefix="/semantic", tags=["semantic"], dependencies=[Depends(require_api_key)])


@router.get("/views", response_model=SemanticCatalogResponse)
async def list_views(
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticCatalogResponse:
    """List available semantic views.

    Parameters
    ----------
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticCatalogResponse
        Catalog response payload.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        payload = await run_in_threadpool(ops.catalog)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            QueryMetrics(
                endpoint="/semantic/views",
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                view_id=None,
                query=None,
                row_count=0,
                truncated=False,
            ),
        )
        raise

    response = SemanticCatalogResponse.model_validate(payload)
    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(
        background,
        QueryMetrics(
            endpoint="/semantic/views",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=None,
            row_count=len(response.views),
            truncated=False,
        ),
    )
    return response


@router.get("/views/{view_id}", response_model=SemanticViewDescriptionResponse)
async def describe_view(
    view_id: str,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticViewDescriptionResponse:
    """Describe a semantic view.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticViewDescriptionResponse
        View description payload.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        payload = await run_in_threadpool(ops.describe, view_id)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            QueryMetrics(
                endpoint=f"/semantic/views/{view_id}",
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                view_id=view_id,
                query=None,
                row_count=0,
                truncated=False,
            ),
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(
        background,
        QueryMetrics(
            endpoint=f"/semantic/views/{view_id}",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=view_id,
            query=None,
            row_count=1,
            truncated=False,
        ),
    )
    return SemanticViewDescriptionResponse.model_validate(payload)


@router.post("/query", response_model=SemanticQueryResponse)
async def query_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticQueryResponse:
    """Execute a semantic query against a view.

    Parameters
    ----------
    payload
        Semantic query request.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticQueryResponse
        Query results.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        response = await run_in_threadpool(ops.query, payload)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            QueryMetrics(
                endpoint="/semantic/query",
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                view_id=payload.view_id,
                query=None,
                row_count=0,
                truncated=False,
            ),
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(
        background,
        QueryMetrics(
            endpoint="/semantic/query",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=len(response.rows),
            truncated=response.truncated,
            query_hash=response.query_hash,
            schema_hash=response.schema_hash,
        ),
    )
    return response


@router.post("/explain", response_model=SemanticExplainResponse)
async def explain_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticExplainResponse:
    """Compile a semantic query and return SQL + plan text.

    Parameters
    ----------
    payload
        Semantic query request.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticExplainResponse
        Compiled SQL plus plan text for the query.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        response = await run_in_threadpool(ops.explain, payload)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            QueryMetrics(
                endpoint="/semantic/explain",
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                view_id=payload.view_id,
                query=None,
                row_count=0,
                truncated=False,
            ),
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(
        background,
        QueryMetrics(
            endpoint="/semantic/explain",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        ),
    )
    return response


__all__ = ["router"]
