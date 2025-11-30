"""Architecture-centric HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FunctionArchitectureResponse, ModuleArchitectureResponse


def build_architecture_router() -> APIRouter:
    """
    Construct the router for architecture and subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing architecture datasets without direct SQL.
    """
    router = APIRouter()

    @router.get(
        "/architecture/function",
        response_model=FunctionArchitectureResponse,
        summary="Get architecture metrics for a function",
    )
    def function_architecture(
        *,
        service: ServiceDep,
        goid_h128: int,
    ) -> FunctionArchitectureResponse:
        """
        Return call-graph architecture metrics for a function.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload for the GOID.

        Raises
        ------
        errors.not_found
            If no architecture row exists for the GOID.
        """
        response = service.get_function_architecture(goid_h128=goid_h128)
        if not response.found or response.architecture is None:
            message = "Function architecture not found"
            raise errors.not_found(message)
        return response

    @router.get(
        "/architecture/module",
        response_model=ModuleArchitectureResponse,
        summary="Get architecture metrics for a module",
    )
    def module_architecture(
        *,
        service: ServiceDep,
        module: str,
    ) -> ModuleArchitectureResponse:
        """
        Return import-graph architecture metrics for a module.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload for the module.

        Raises
        ------
        errors.not_found
            If no architecture row exists for the module.
        """
        response = service.get_module_architecture(module=module)
        if not response.found or response.architecture is None:
            message = "Module architecture not found"
            raise errors.not_found(message)
        return response

    return router


__all__ = ["build_architecture_router"]
