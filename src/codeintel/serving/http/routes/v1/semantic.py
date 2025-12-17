"""V1 semantic HTTP routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import get_kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)

router = APIRouter(prefix="/semantic", tags=["semantic"], dependencies=[Depends(require_api_key)])

_KERNEL_DEPENDENCY = Depends(get_kernel)


@router.get("/views", response_model=SemanticCatalogResponse)
async def list_views(
    background: BackgroundTasks,
    request: Request,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> SemanticCatalogResponse:
    """List available semantic views.

    Parameters
    ----------
    kernel
        Semantic query kernel.
    background
        Background task queue.
    request
        Current HTTP request.

    Returns
    -------
    SemanticCatalogResponse
        Catalog response payload.

    Raises
    ------
    TypeError
        When FastAPI fails to inject required dependencies.
    """
    if not isinstance(background, BackgroundTasks):
        msg = "FastAPI did not provide a BackgroundTasks instance"
        raise TypeError(msg)
    if not isinstance(request, Request):
        msg = "FastAPI did not provide a Request instance"
        raise TypeError(msg)
    if not isinstance(kernel, SemanticQueryKernel):
        msg = "FastAPI did not provide a SemanticQueryKernel instance"
        raise TypeError(msg)
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    payload = await run_in_threadpool(kernel.catalog)
    response = SemanticCatalogResponse.model_validate(payload)
    duration_ms = (time.perf_counter() - start) * 1000
    background.add_task(
        log_query_metrics,
        QueryMetrics(
            endpoint="/semantic/views",
            view_id=None,
            query=None,
            row_count=len(response.views),
            truncated=False,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        ),
    )
    return response


@router.get("/views/{view_id}", response_model=SemanticViewDescriptionResponse)
async def describe_view(
    view_id: str,
    background: BackgroundTasks,
    request: Request,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> SemanticViewDescriptionResponse:
    """Describe a semantic view.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    kernel
        Semantic query kernel.
    background
        Background task queue.
    request
        Current HTTP request.

    Returns
    -------
    SemanticViewDescriptionResponse
        View description payload.

    Raises
    ------
    ServingError
        When the view ID does not exist.
    TypeError
        When FastAPI fails to inject required dependencies.
    """
    if not isinstance(background, BackgroundTasks):
        msg = "FastAPI did not provide a BackgroundTasks instance"
        raise TypeError(msg)
    if not isinstance(request, Request):
        msg = "FastAPI did not provide a Request instance"
        raise TypeError(msg)
    if not isinstance(kernel, SemanticQueryKernel):
        msg = "FastAPI did not provide a SemanticQueryKernel instance"
        raise TypeError(msg)
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        payload = await run_in_threadpool(kernel.describe, view_id)
    except KeyError as exc:
        raise ServingError(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=str(exc),
        ) from exc
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        background.add_task(
            log_query_metrics,
            QueryMetrics(
                endpoint=f"/semantic/views/{view_id}",
                view_id=view_id,
                query=None,
                row_count=1,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
        )
    return SemanticViewDescriptionResponse.model_validate(payload)


@router.post("/query", response_model=SemanticQueryResponse)
async def query_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> SemanticQueryResponse:
    """Execute a semantic query against a view.

    Parameters
    ----------
    payload
        Semantic query request.
    kernel
        Semantic query kernel.
    background
        Background task queue.
    request
        Current HTTP request.

    Returns
    -------
    SemanticQueryResponse
        Query results.

    Raises
    ------
    ServingError
        When the view is missing or the request is invalid.
    TypeError
        When FastAPI fails to inject a SemanticQueryRequest.
    RuntimeError
        Internal error if response is unexpectedly None (should never happen).
    """
    if not isinstance(payload, SemanticQueryRequest):
        msg = "FastAPI did not provide a SemanticQueryRequest model"
        raise TypeError(msg)
    if not isinstance(background, BackgroundTasks):
        msg = "FastAPI did not provide a BackgroundTasks instance"
        raise TypeError(msg)
    if not isinstance(request, Request):
        msg = "FastAPI did not provide a Request instance"
        raise TypeError(msg)
    if not isinstance(kernel, SemanticQueryKernel):
        msg = "FastAPI did not provide a SemanticQueryKernel instance"
        raise TypeError(msg)
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    response: SemanticQueryResponse | None = None
    try:
        response = await run_in_threadpool(kernel.query, payload)
    except KeyError as exc:
        raise ServingError(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=str(exc),
        ) from exc
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
                endpoint="/semantic/query",
                view_id=payload.view_id,
                query=None,
                row_count=len(response.rows) if response else 0,
                truncated=response.truncated if response else False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
        )
    # Unreachable if exception was raised; type narrowing for checker
    if response is None:  # pragma: no cover
        msg = "Unexpected state: response not set"
        raise RuntimeError(msg)
    return response


@router.post("/explain", response_model=SemanticExplainResponse)
async def explain_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> SemanticExplainResponse:
    """Compile a semantic query and return SQL + plan text.

    Parameters
    ----------
    payload
        Semantic query request.
    kernel
        Semantic query kernel.
    background
        Background task queue.
    request
        Current HTTP request.

    Returns
    -------
    SemanticExplainResponse
        Compiled SQL plus plan text for the query.

    Raises
    ------
    ServingError
        When the view is missing or the request is invalid.
    TypeError
        When FastAPI fails to inject a SemanticQueryRequest.
    """
    if not isinstance(payload, SemanticQueryRequest):
        msg = "FastAPI did not provide a SemanticQueryRequest model"
        raise TypeError(msg)
    if not isinstance(background, BackgroundTasks):
        msg = "FastAPI did not provide a BackgroundTasks instance"
        raise TypeError(msg)
    if not isinstance(request, Request):
        msg = "FastAPI did not provide a Request instance"
        raise TypeError(msg)
    if not isinstance(kernel, SemanticQueryKernel):
        msg = "FastAPI did not provide a SemanticQueryKernel instance"
        raise TypeError(msg)
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        response = await run_in_threadpool(kernel.explain, payload)
    except KeyError as exc:
        raise ServingError(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=str(exc),
        ) from exc
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
                endpoint="/semantic/explain",
                view_id=payload.view_id,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
            ),
        )
    return response


__all__ = ["router"]
