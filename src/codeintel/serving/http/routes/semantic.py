"""Semantic layer HTTP endpoints.

Provides REST API access to the semantic query kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException

from codeintel.serving.semantic.models import SemanticQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.semantic.kernel import SemanticQueryKernel

router = APIRouter(prefix="/semantic", tags=["semantic"])


def get_kernel() -> SemanticQueryKernel:
    """Return the SemanticQueryKernel from application wiring."""
    msg = "get_kernel must be overridden by app wiring"
    raise NotImplementedError(msg)


_KERNEL_DEPENDENCY = Depends(get_kernel)
_QUERY_BODY = Body(...)


@router.get("/views")
async def list_views(kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY) -> dict[str, object]:
    """List available semantic views.

    Returns
    -------
    dict[str, object]
        Catalog response payload.
    """
    return kernel.catalog()


@router.get("/views/{view_id}")
async def describe_view(
    view_id: str, kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY
) -> dict[str, object]:
    """Describe a semantic view.

    Returns
    -------
    dict[str, object]
        View description payload.

    Raises
    ------
    HTTPException
        If the requested view does not exist.
    """
    try:
        return kernel.describe(view_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/query")
async def query_view(
    payload: dict[str, object] = _QUERY_BODY,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> dict[str, object]:
    """Execute a semantic view query.

    Returns
    -------
    dict[str, object]
        Query response payload.

    Raises
    ------
    HTTPException
        If the view does not exist or the request payload is invalid.
    """
    try:
        request = SemanticQueryRequest.model_validate(payload)
        return kernel.query(request).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/explain")
async def explain_view(
    payload: dict[str, object] = _QUERY_BODY,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> dict[str, object]:
    """Return compiled SQL and DuckDB plan for a semantic query.

    Returns
    -------
    dict[str, object]
        Explain response payload.

    Raises
    ------
    HTTPException
        If the view does not exist or the request payload is invalid.
    """
    try:
        request = SemanticQueryRequest.model_validate(payload)
        return kernel.explain(request).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
