"""Shared typed dependencies for serving HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from codeintel.serving.auth.policy import require_http_auth
from codeintel.serving.errors import AuthForbiddenError
from codeintel.serving.http.state import ServingState
from codeintel.serving.operations.ops import ServingOperations
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
    """Extract the SemanticQueryKernel from the serving state.

    Parameters
    ----------
    state
        Application serving state.

    Returns
    -------
    SemanticQueryKernel
        The kernel instance for semantic queries.
    """
    return state.kernel


Kernel = Annotated[SemanticQueryKernel, Depends(get_kernel)]


def get_ops(state: State) -> ServingOperations:
    """Extract the ServingOperations facade from the serving state.

    Parameters
    ----------
    state
        Application serving state.

    Returns
    -------
    ServingOperations
        Operations facade for transport adapters.
    """
    return state.ops


Ops = Annotated[ServingOperations, Depends(get_ops)]


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
    AuthForbiddenError
        When an API key is configured and missing/invalid.
    """
    try:
        require_http_auth(headers=request.headers, settings=state.settings)
    except AuthForbiddenError as exc:
        raise AuthForbiddenError(reason=exc.details.get("reason")) from exc


__all__ = ["Kernel", "Ops", "State", "get_kernel", "get_ops", "require_api_key"]
