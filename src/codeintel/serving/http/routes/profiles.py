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


def build_profiles_router() -> APIRouter:
    """
    Construct the router for profile endpoints.

    Returns
    -------
    APIRouter
        Router exposing function, file, and module profiles.
    """
    router = APIRouter()

    @router.get(
        "/profiles/function",
        response_model=FunctionProfileResponse,
        summary="Get a function profile",
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
        profile = service.get_function_profile(goid_h128=goid_h128)
        if not profile.found or profile.profile is None:
            message = "Function profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        "/profiles/file",
        response_model=FileProfileResponse,
        summary="Get a file profile",
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
        profile = service.get_file_profile(rel_path=rel_path)
        if not profile.found or profile.profile is None:
            message = "File profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        "/profiles/module",
        response_model=ModuleProfileResponse,
        summary="Get a module profile",
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
        profile = service.get_module_profile(module=module)
        if not profile.found or profile.profile is None:
            message = "Module profile not found"
            raise errors.not_found(message)
        return profile

    return router


__all__ = ["build_profiles_router"]
