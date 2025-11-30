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
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def _load_subsystem_specs() -> tuple[dict[str, OperationSpec], dict[str, str]]:
    ids = [
        "subsystems.list",
        "subsystems.profiles",
        "subsystems.coverage",
        "subsystems.module_memberships",
        "subsystems.detail",
    ]
    specs: dict[str, OperationSpec] = {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    missing_paths: list[str] = []
    for op_id in ids:
        spec = get_operation_spec(op_id)
        if spec is None:
            missing.append(op_id)
            continue
        specs[op_id] = spec
        if spec.http_path is None:
            missing_paths.append(op_id)
        else:
            paths[op_id] = spec.http_path
    if missing or missing_paths:
        message = (
            f"Missing OperationSpec entries: {missing or 'ok'}; paths: {missing_paths or 'ok'}"
        )
        raise ValueError(message)
    return specs, paths


def build_subsystem_router() -> APIRouter:
    """
    Construct the router for subsystem endpoints.

    Raises
    ------
    ValueError
        If OperationSpec entries are missing or lack paths.

    Returns
    -------
    APIRouter
        Router exposing subsystem docs views and membership helpers.
    """
    router = APIRouter()
    try:
        specs, paths = _load_subsystem_specs()
    except ValueError as exc:
        message = "Failed to load subsystem OperationSpec entries"
        raise ValueError(message) from exc
    spec_list = specs["subsystems.list"]
    spec_profiles = specs["subsystems.profiles"]
    spec_coverage = specs["subsystems.coverage"]
    spec_memberships = specs["subsystems.module_memberships"]
    spec_detail = specs["subsystems.detail"]
    list_path = paths["subsystems.list"]
    profiles_path = paths["subsystems.profiles"]
    coverage_path = paths["subsystems.coverage"]
    memberships_path = paths["subsystems.module_memberships"]
    detail_path = paths["subsystems.detail"]

    @router.get(
        list_path,
        response_model=SubsystemSummaryResponse,
        summary=spec_list.summary,
        tags=[spec_list.category],
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
        domain_response = service.list_subsystems(limit=limit, role=role, q=q)
        return SubsystemSummaryResponse.from_domain(domain_response)

    @router.get(
        profiles_path,
        response_model=SubsystemProfileResponse,
        summary=spec_profiles.summary,
        tags=[spec_profiles.category],
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
        domain_response = service.list_subsystem_profiles(limit=limit)
        return SubsystemProfileResponse.from_domain(domain_response)

    @router.get(
        coverage_path,
        response_model=SubsystemCoverageResponse,
        summary=spec_coverage.summary,
        tags=[spec_coverage.category],
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
        domain_response = service.list_subsystem_coverage(limit=limit)
        return SubsystemCoverageResponse.from_domain(domain_response)

    @router.get(
        memberships_path,
        response_model=ModuleSubsystemResponse,
        summary=spec_memberships.summary,
        tags=[spec_memberships.category],
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
        domain_response = service.get_module_subsystems(module=module)
        response = ModuleSubsystemResponse.from_domain(domain_response)
        if not response.found or not response.memberships:
            message = "Module has no subsystem mappings"
            raise errors.not_found(message)
        return response

    @router.get(
        detail_path,
        response_model=SubsystemModulesResponse,
        summary=spec_detail.summary,
        tags=[spec_detail.category],
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
        domain_response = service.get_subsystem_modules(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        response = SubsystemModulesResponse.from_domain(domain_response)
        if not response.found or response.subsystem is None:
            message = "Subsystem not found"
            raise errors.not_found(message)
        return response

    return router


__all__ = ["build_subsystem_router"]
