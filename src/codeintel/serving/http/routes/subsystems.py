"""Subsystem HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSummaryResponse,
)


def build_subsystem_router() -> APIRouter:
    """
    Construct the router for subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing subsystem docs views and membership helpers.
    """
    router = APIRouter()

    @router.get(
        "/architecture/subsystems",
        response_model=SubsystemSummaryResponse,
        summary="List inferred subsystems",
    )
    def list_subsystems(
        *,
        service: ServiceDep,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        """
        List inferred subsystems derived from module coupling.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        return service.list_subsystems(limit=limit, role=role, q=q)

    @router.get(
        "/architecture/subsystem-profiles",
        response_model=SubsystemProfileResponse,
        summary="List subsystem profiles",
    )
    def list_subsystem_profiles(
        *,
        service: ServiceDep,
        limit: int | None = None,
    ) -> SubsystemProfileResponse:
        """
        List subsystem profiles backed by docs views.

        Returns
        -------
        SubsystemProfileResponse
            Profile rows with metadata.
        """
        return service.list_subsystem_profiles(limit=limit)

    @router.get(
        "/architecture/subsystem-coverage",
        response_model=SubsystemCoverageResponse,
        summary="List subsystem coverage rollups",
    )
    def list_subsystem_coverage(
        *,
        service: ServiceDep,
        limit: int | None = None,
    ) -> SubsystemCoverageResponse:
        """
        List subsystem coverage rollups derived from test profiles.

        Returns
        -------
        SubsystemCoverageResponse
            Coverage rows with metadata.
        """
        return service.list_subsystem_coverage(limit=limit)

    @router.get(
        "/architecture/module-subsystems",
        response_model=ModuleSubsystemResponse,
        summary="List subsystem memberships for a module",
    )
    def module_subsystems(
        *,
        service: ServiceDep,
        module: str,
    ) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for the requested module.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.

        Raises
        ------
        errors.not_found
            If the module is not mapped to any subsystem.
        """
        response = service.get_module_subsystems(module=module)
        if not response.found or not response.memberships:
            message = "Module has no subsystem mappings"
            raise errors.not_found(message)
        return response

    @router.get(
        "/architecture/subsystem",
        response_model=SubsystemModulesResponse,
        summary="Get modules and detail for a subsystem",
    )
    def subsystem_modules(
        *,
        service: ServiceDep,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        """
        Return subsystem metadata and modules.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail and member modules.

        Raises
        ------
        errors.not_found
            If the subsystem cannot be located.
        """
        response = service.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        if not response.found or response.subsystem is None:
            message = "Subsystem not found"
            raise errors.not_found(message)
        return response

    return router


__all__ = ["build_subsystem_router"]
