"""Shared typed dependencies for serving HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.state import ServingState
from codeintel.serving.semantic.kernel import SemanticQueryKernel


def _get_state(request: Request) -> ServingState:
    if not isinstance(request, Request):
        msg = "FastAPI did not provide a Request instance"
        raise TypeError(msg)

    raw = getattr(request.app.state, "serving", None)
    if isinstance(raw, ServingState):
        return raw
    msg = "ServingState not configured on app.state"
    raise RuntimeError(msg)


State = Annotated[ServingState, Depends(_get_state)]


def get_kernel(state: State) -> SemanticQueryKernel:
    return state.kernel


Kernel = Annotated[SemanticQueryKernel, Depends(get_kernel)]


def require_api_key(request: Request, state: State) -> None:
    """Optionally require a valid API key for this request.

    Parameters
    ----------
    request
        Current request.
    state
        Application serving state.

    Raises
    ------
    ServingError
        When an API key is configured and missing/invalid.
    """
    expected = state.settings.api_key
    if expected is None:
        return

    provided = request.headers.get("X-API-Key")
    if provided == expected:
        return

    raise ServingError(
        problem_type=ProblemType.UNAUTHORIZED,
        title="Unauthorized",
        status=401,
        detail="Invalid or missing API key.",
    )


__all__ = ["Kernel", "State", "get_kernel", "require_api_key"]
