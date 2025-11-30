"""IDE-centric HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FileHintsResponse


def build_ide_router() -> APIRouter:
    """
    Construct the router for IDE-facing hint endpoints.

    Returns
    -------
    APIRouter
        Router exposing contextual hints for editor integrations.
    """
    router = APIRouter()

    @router.get(
        "/ide/hints",
        response_model=FileHintsResponse,
        summary="Get IDE hints for a file",
    )
    def ide_hints(
        *,
        service: ServiceDep,
        rel_path: str,
    ) -> FileHintsResponse:
        """
        Return subsystem and module context suitable for IDE tooltips.

        Returns
        -------
        FileHintsResponse
            Hint rows keyed by the provided relative path.

        Raises
        ------
        errors.not_found
            If no hints can be derived for the path.
        """
        response = service.get_file_hints(rel_path=rel_path)
        if not response.found or not response.hints:
            message = "IDE hints not found for path"
            raise errors.not_found(message)
        return response

    return router


__all__ = ["build_ide_router"]
