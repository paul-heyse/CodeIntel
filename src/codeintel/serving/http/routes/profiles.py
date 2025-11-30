"""Profile HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionProfileResponse,
    ModuleProfileResponse,
)
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_profiles_router() -> APIRouter:
    """
    Construct the router for profile endpoints.

    Raises
    ------
    ValueError
        If OperationSpec entries are missing or lack http_path values.

    Returns
    -------
    APIRouter
        Router exposing function, file, and module profiles.
    """
    router = APIRouter()
    spec_function = _require_spec("profiles.function")
    spec_file = _require_spec("profiles.file")
    spec_module = _require_spec("profiles.module")
    if spec_function.http_path is None or spec_file.http_path is None or spec_module.http_path is None:
        message = "Profile OperationSpec entries must define http_path"
        raise ValueError(message)
    function_path = spec_function.http_path
    file_path = spec_file.http_path
    module_path = spec_module.http_path

    @router.get(
        function_path,
        response_model=FunctionProfileResponse,
        summary=spec_function.summary,
        tags=[spec_function.category],
    )
    def function_profile(
        *,
        service: ServiceDep,
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
    )
    def file_profile(
        *,
        service: ServiceDep,
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
    )
    def module_profile(
        *,
        service: ServiceDep,
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
