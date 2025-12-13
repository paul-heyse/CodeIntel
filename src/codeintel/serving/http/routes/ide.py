"""IDE-centric HTTP routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from codeintel.serving.http import dependencies as http_deps
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FileHintsResponse
from codeintel.serving.operations import get_operation

if TYPE_CHECKING:
    from codeintel.serving.http.routes.functions import RouterOptions
    from codeintel.serving.operations import Operation


def _require_spec(op_id: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None:
        message = f"Operation {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_ide_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for IDE-facing hint endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites.

    Raises
    ------
    ValueError
        If the Operation for IDE hints is missing or incomplete.

    Returns
    -------
    APIRouter
        Router exposing contextual hints for editor integrations.
    """
    router = APIRouter()
    spec = _require_spec("ide.hints")
    if spec.http_path is None:
        message = "Operation ide.hints is missing http_path"
        raise ValueError(message)
    path = spec.http_path

    auto_pipeline = options is not None and options.auto_pipeline
    ide_deps = (
        [Depends(http_deps.make_op_prereq_dependency("ide.hints"))] if auto_pipeline else []
    )

    @router.get(
        path,
        response_model=FileHintsResponse,
        summary=spec.summary,
        tags=[spec.category],
        dependencies=ide_deps,
    )
    def ide_hints(
        *,
        service: http_deps.ServiceDep,
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
        result = service.get_file_hints(rel_path=rel_path)
        if not result.found or not result.hints:
            message = "IDE hints not found for path"
            raise errors.not_found(message)
        return FileHintsResponse.from_domain(result)

    return router


__all__ = ["build_ide_router"]
