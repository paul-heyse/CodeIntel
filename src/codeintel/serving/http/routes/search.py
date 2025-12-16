"""Code metadata search HTTP endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Body, Depends, HTTPException

from codeintel.serving.search.models import SearchQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.semantic.kernel import SemanticQueryKernel

router = APIRouter(prefix="/search", tags=["search"])


def get_kernel() -> SemanticQueryKernel:
    """Return the SemanticQueryKernel from application wiring."""
    msg = "get_kernel must be overridden by app wiring"
    raise NotImplementedError(msg)


_KERNEL_DEPENDENCY = Depends(get_kernel)
_SEARCH_BODY = Body(...)


@router.post("")
async def search(
    payload: dict[str, object] = _SEARCH_BODY,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> dict[str, object]:
    """Search code metadata for the current serving snapshot.

    Returns
    -------
    dict[str, object]
        Search response payload.

    Raises
    ------
    HTTPException
        If the request payload is invalid.
    """
    try:
        request = SearchQueryRequest.model_validate(payload)
        return kernel.search(request).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
