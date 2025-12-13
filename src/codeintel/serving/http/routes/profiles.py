"""Profile HTTP routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from codeintel.serving.http import dependencies as http_deps
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionProfileResponse,
    ModuleProfileResponse,
)
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


def build_profiles_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for profile endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites
        before the operation executes.

    Raises
    ------
    ValueError
        If Operation entries are missing or lack http_path values.

    Returns
    -------
    APIRouter
        Router exposing function, file, and module profiles.
    """
    router = APIRouter()
    spec_function = _require_spec("profiles.function")
    spec_file = _require_spec("profiles.file")
    spec_module = _require_spec("profiles.module")
    if (
        spec_function.http_path is None
        or spec_file.http_path is None
        or spec_module.http_path is None
    ):
        message = "Profile Operation entries must define http_path"
        raise ValueError(message)
    function_path = spec_function.http_path
    file_path = spec_file.http_path
    module_path = spec_module.http_path

    auto_pipeline = options is not None and options.auto_pipeline
    func_deps = (
        [Depends(http_deps.make_op_prereq_dependency("profiles.function"))]
        if auto_pipeline
        else []
    )
    file_deps = (
        [Depends(http_deps.make_op_prereq_dependency("profiles.file"))] if auto_pipeline else []
    )
    module_deps = (
        [Depends(http_deps.make_op_prereq_dependency("profiles.module"))] if auto_pipeline else []
    )

    @router.get(
        function_path,
        response_model=FunctionProfileResponse,
        summary=spec_function.summary,
        tags=[spec_function.category],
        dependencies=func_deps,
    )
    def function_profile(
        *,
        service: http_deps.ServiceDep,
        goid_h128: int,
    ) -> FunctionProfileResponse:
        """
        Return a denormalized function profile for the given GOID.

        Returns
        -------
        FunctionProfileResponse
            Profile payload for the requested GOID.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        domain_profile = service.get_function_profile(goid_h128=goid_h128)
        profile = FunctionProfileResponse.from_domain(domain_profile)
        if not profile.found or profile.profile is None:
            message = "Function profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        file_path,
        response_model=FileProfileResponse,
        summary=spec_file.summary,
        tags=[spec_file.category],
        dependencies=file_deps,
    )
    def file_profile(
        *,
        service: http_deps.ServiceDep,
        rel_path: str,
    ) -> FileProfileResponse:
        """
        Return a denormalized profile for a file path.

        Returns
        -------
        FileProfileResponse
            Profile payload for the requested file.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        domain_profile = service.get_file_profile(rel_path=rel_path)
        profile = FileProfileResponse.from_domain(domain_profile)
        if not profile.found or profile.profile is None:
            message = "File profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        module_path,
        response_model=ModuleProfileResponse,
        summary=spec_module.summary,
        tags=[spec_module.category],
        dependencies=module_deps,
    )
    def module_profile(
        *,
        service: http_deps.ServiceDep,
        module: str,
    ) -> ModuleProfileResponse:
        """
        Return a module-level profile including coverage and import metrics.

        Returns
        -------
        ModuleProfileResponse
            Profile payload for the requested module.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        domain_profile = service.get_module_profile(module=module)
        profile = ModuleProfileResponse.from_domain(domain_profile)
        if not profile.found or profile.profile is None:
            message = "Module profile not found"
            raise errors.not_found(message)
        return profile

    return router


__all__ = ["build_profiles_router"]
