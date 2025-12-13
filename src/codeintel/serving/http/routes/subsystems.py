"""Subsystem HTTP routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends

from codeintel.serving.http import dependencies as http_deps
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSummaryResponse,
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


def _load_subsystem_specs() -> tuple[dict[str, Operation], dict[str, str]]:
    ids = [
        "subsystems.list",
        "subsystems.profiles",
        "subsystems.coverage",
        "subsystems.module_memberships",
        "subsystems.detail",
    ]
    specs: dict[str, Operation] = {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    missing_paths: list[str] = []
    for op_id in ids:
        spec = get_operation(op_id)
        if spec is None:
            missing.append(op_id)
            continue
        specs[op_id] = spec
        if spec.http_path is None:
            missing_paths.append(op_id)
        else:
            paths[op_id] = spec.http_path
    if missing or missing_paths:
        message = f"Missing Operation entries: {missing or 'ok'}; paths: {missing_paths or 'ok'}"
        raise ValueError(message)
    return specs, paths


RouteDeps = Sequence[Any]


def _build_prereq_deps(
    specs: dict[str, Operation],
    options: RouterOptions | None,
) -> dict[str, RouteDeps]:
    """Build auto-pipeline dependencies for each operation.

    Parameters
    ----------
    specs
        Operation specifications keyed by operation ID.
    options
        Router options with auto_pipeline flag.

    Returns
    -------
    dict[str, RouteDeps]
        Mapping of operation ID to dependency list.
    """
    if options is None or not options.auto_pipeline:
        return {}
    return {op_id: [Depends(http_deps.make_op_prereq_dependency(op_id))] for op_id in specs}


def build_subsystem_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for subsystem endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites.

    Raises
    ------
    ValueError
        If Operation entries are missing or lack paths.

    Returns
    -------
    APIRouter
        Router exposing subsystem docs views and membership helpers.
    """
    router = APIRouter()
    try:
        specs, paths = _load_subsystem_specs()
    except ValueError as exc:
        message = "Failed to load subsystem Operation entries"
        raise ValueError(message) from exc

    deps = _build_prereq_deps(specs, options)
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
        dependencies=list(deps.get("subsystems.list", [])),
    )
    def list_subsystems(
        *,
        service: http_deps.ServiceDep,
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
        dependencies=list(deps.get("subsystems.profiles", [])),
    )
    def list_subsystem_profiles(
        *,
        service: http_deps.ServiceDep,
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
        dependencies=list(deps.get("subsystems.coverage", [])),
    )
    def list_subsystem_coverage(
        *,
        service: http_deps.ServiceDep,
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
        dependencies=list(deps.get("subsystems.module_memberships", [])),
    )
    def module_subsystems(
        *,
        service: http_deps.ServiceDep,
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
        dependencies=list(deps.get("subsystems.detail", [])),
    )
    def subsystem_modules(
        *,
        service: http_deps.ServiceDep,
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
