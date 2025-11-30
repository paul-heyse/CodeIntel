"""IDE-centric HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FileHintsResponse
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_ide_router() -> APIRouter:
    """
    Construct the router for IDE-facing hint endpoints.

    Raises
    ------
    ValueError
        If the OperationSpec for IDE hints is missing or incomplete.

    Returns
    -------
    APIRouter
        Router exposing contextual hints for editor integrations.
    """
    router = APIRouter()
    spec = _require_spec("ide.hints")
    if spec.http_path is None:
        message = "OperationSpec ide.hints is missing http_path"
        raise ValueError(message)
    path = spec.http_path

    @router.get(
        path,
        response_model=FileHintsResponse,
        summary=spec.summary,
        tags=[spec.category],
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
