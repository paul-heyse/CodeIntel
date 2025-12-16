"""V1 semantic HTTP routes."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)

router = APIRouter(prefix="/semantic", tags=["semantic"], dependencies=[Depends(require_api_key)])


@router.get("/views", response_model=SemanticCatalogResponse)
async def list_views(kernel: Kernel) -> SemanticCatalogResponse:
    """List available semantic views.

    Returns
    -------
    SemanticCatalogResponse
        Catalog response payload.
    """
    kernel = cast("Kernel", kernel)
    payload = await run_in_threadpool(kernel.catalog)
    return SemanticCatalogResponse.model_validate(payload)


@router.get("/views/{view_id}", response_model=SemanticViewDescriptionResponse)
async def describe_view(view_id: str, kernel: Kernel) -> SemanticViewDescriptionResponse:
    """Describe a semantic view.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    kernel
        Semantic query kernel.

    Returns
    -------
    SemanticViewDescriptionResponse
        View description payload.

    Raises
    ------
    ServingError
        When the view ID does not exist.
    """
    kernel = cast("Kernel", kernel)
    try:
        payload = await run_in_threadpool(kernel.describe, view_id)
        return SemanticViewDescriptionResponse.model_validate(payload)
    except KeyError as exc:
        raise ServingError(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=str(exc),
        ) from exc


@router.post("/query", response_model=SemanticQueryResponse)
async def query_view(payload: SemanticQueryRequest, kernel: Kernel) -> SemanticQueryResponse:
    """Execute a semantic query against a view.

    Parameters
    ----------
    payload
        Semantic query request.
    kernel
        Semantic query kernel.

    Returns
    -------
    SemanticQueryResponse
        Query results.

    Raises
    ------
    ServingError
        When the view is missing or the request is invalid.
    """
    if not isinstance(payload, SemanticQueryRequest):
        msg = "FastAPI did not provide a SemanticQueryRequest model"
        raise TypeError(msg)
    kernel = cast("Kernel", kernel)
    try:
        return await run_in_threadpool(kernel.query, payload)
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


@router.post("/explain", response_model=SemanticExplainResponse)
async def explain_view(payload: SemanticQueryRequest, kernel: Kernel) -> SemanticExplainResponse:
    """Compile a semantic query and return SQL + plan text.

    Parameters
    ----------
    payload
        Semantic query request.
    kernel
        Semantic query kernel.

    Returns
    -------
    SemanticExplainResponse
        Compiled SQL plus plan text for the query.

    Raises
    ------
    ServingError
        When the view is missing or the request is invalid.
    """
    if not isinstance(payload, SemanticQueryRequest):
        msg = "FastAPI did not provide a SemanticQueryRequest model"
        raise TypeError(msg)
    kernel = cast("Kernel", kernel)
    try:
        return await run_in_threadpool(kernel.explain, payload)
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


__all__ = ["router"]
