"""Architecture-centric HTTP routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from codeintel.serving.http.dependencies import ServiceDep, make_op_prereq_dependency
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import FunctionArchitectureResponse, ModuleArchitectureResponse
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


def build_architecture_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for architecture and subsystem endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites.

    Raises
    ------
    ValueError
        If Operation entries are missing or lack http_path values.

    Returns
    -------
    APIRouter
        Router exposing architecture datasets without direct SQL.
    """
    router = APIRouter()
    spec_function = _require_spec("architecture.function")
    spec_module = _require_spec("architecture.module")
    if spec_function.http_path is None:
        message = "Operation architecture.function is missing http_path"
        raise ValueError(message)
    function_path = spec_function.http_path
    if spec_module.http_path is None:
        message = "Operation architecture.module is missing http_path"
        raise ValueError(message)
    module_path = spec_module.http_path

    auto_pipeline = options is not None and options.auto_pipeline
    func_deps = (
        [Depends(make_op_prereq_dependency("architecture.function"))] if auto_pipeline else []
    )
    mod_deps = [Depends(make_op_prereq_dependency("architecture.module"))] if auto_pipeline else []

    @router.get(
        function_path,
        response_model=FunctionArchitectureResponse,
        summary=spec_function.summary,
        tags=[spec_function.category],
        dependencies=func_deps,
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
        result = service.get_function_architecture(goid_h128=goid_h128)
        if not result.found or result.architecture is None:
            message = "Function architecture not found"
            raise errors.not_found(message)
        return FunctionArchitectureResponse.from_domain(result)

    @router.get(
        module_path,
        response_model=ModuleArchitectureResponse,
        summary=spec_module.summary,
        tags=[spec_module.category],
        dependencies=mod_deps,
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
        result = service.get_module_architecture(module=module)
        if not result.found or result.architecture is None:
            message = "Module architecture not found"
            raise errors.not_found(message)
        return ModuleArchitectureResponse.from_domain(result)

    return router


__all__ = ["build_architecture_router"]
