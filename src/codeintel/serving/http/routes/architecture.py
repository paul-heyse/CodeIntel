"""Architecture-centric HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FunctionArchitectureResponse, ModuleArchitectureResponse
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_architecture_router() -> APIRouter:
    """
    Construct the router for architecture and subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing architecture datasets without direct SQL.
    """
    router = APIRouter()
    spec_function = _require_spec("architecture.function")
    spec_module = _require_spec("architecture.module")

    @router.get(
        spec_function.http_path,
        response_model=FunctionArchitectureResponse,
        summary=spec_function.summary,
        tags=[spec_function.category],
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
        spec_module.http_path,
        response_model=ModuleArchitectureResponse,
        summary=spec_module.summary,
        tags=[spec_module.category],
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
