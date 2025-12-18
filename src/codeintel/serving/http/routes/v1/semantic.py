"""V1 semantic HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.metrics import QueryMetrics
from codeintel.serving.http.route_utils import run_in_threadpool_with_metrics
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
    def _success(payload: dict[str, object], duration_ms: float, correlation_id: str) -> QueryMetrics:
        views_obj = payload.get("views")
        views = views_obj if isinstance(views_obj, list) else []
        return QueryMetrics(
            endpoint="/semantic/views",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=None,
            row_count=len(views),
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/semantic/views",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=None,
            row_count=0,
            truncated=False,
        )

    payload = await run_in_threadpool_with_metrics(background, request, ops.catalog, _success, _error)
    return SemanticCatalogResponse.model_validate(payload)


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
    def _success(_payload: dict[str, object], duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint=f"/semantic/views/{view_id}",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=view_id,
            query=None,
            row_count=1,
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint=f"/semantic/views/{view_id}",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    payload = await run_in_threadpool_with_metrics(background, request, ops.describe, _success, _error, view_id)
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
    def _success(response: SemanticQueryResponse, duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/semantic/query",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=len(response.rows),
            truncated=response.truncated,
            query_hash=response.query_hash,
            schema_hash=response.schema_hash,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/semantic/query",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    return await run_in_threadpool_with_metrics(background, request, ops.query, _success, _error, payload)


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
    def _success(_response: SemanticExplainResponse, duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/semantic/explain",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/semantic/explain",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    return await run_in_threadpool_with_metrics(background, request, ops.explain, _success, _error, payload)


__all__ = ["router"]
