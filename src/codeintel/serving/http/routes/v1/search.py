"""V1 search HTTP routes."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchQueryResponse)
async def search(payload: SearchQueryRequest, kernel: Kernel) -> SearchQueryResponse:
    """Search code metadata for the current serving snapshot.

    Parameters
    ----------
    payload
        Search request payload.
    kernel
        Semantic query kernel.

    Returns
    -------
    SearchQueryResponse
        Search results.

    Raises
    ------
    ServingError
        When the request payload is invalid.
    """
    if not isinstance(payload, SearchQueryRequest):
        msg = "FastAPI did not provide a SearchQueryRequest model"
        raise TypeError(msg)
    kernel = cast("Kernel", kernel)
    try:
        return await run_in_threadpool(kernel.search, payload)
    except ValueError as exc:
        raise ServingError(
            problem_type=ProblemType.INVALID_QUERY,
            title="Invalid Query",
            status=400,
            detail=str(exc),
        ) from exc


__all__ = ["router"]
