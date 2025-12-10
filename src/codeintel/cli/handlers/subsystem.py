"""Subsystem exploration handlers following the unified handler pattern.

This module provides handlers for subsystem commands using the
HandlerContext pattern for consistent resource management
and output rendering.

All handlers in this module:

1. Accept HandlerContext as their only argument
2. Return CliResult[T]
3. Never write to stdout/stderr directly
4. Never call sys.exit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers.context import HandlerContext
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import (
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)

LOG = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True)
class SubsystemListResult:
    """Result from subsystem list operation.

    Parameters
    ----------
    subsystems
        List of subsystem data.
    meta
        Response metadata.
    """

    subsystems: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "subsystems": self.subsystems,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemShowResult:
    """Result from subsystem show operation.

    Parameters
    ----------
    subsystem
        Subsystem detail data.
    modules
        List of module data in the subsystem.
    meta
        Response metadata.
    """

    subsystem: dict[str, Any]
    modules: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "subsystem": self.subsystem,
            "modules": self.modules,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemProfilesResult:
    """Result from subsystem profiles operation.

    Parameters
    ----------
    profiles
        List of profile data.
    meta
        Response metadata.
    """

    profiles: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "profiles": self.profiles,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemCoverageResult:
    """Result from subsystem coverage operation.

    Parameters
    ----------
    coverage
        List of coverage data.
    meta
        Response metadata.
    """

    coverage: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "coverage": self.coverage,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemMembershipResult:
    """Result from subsystem membership operation.

    Parameters
    ----------
    found
        Whether the module was found.
    memberships
        List of membership data.
    meta
        Response metadata.
    """

    found: bool
    memberships: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "found": self.found,
            "memberships": self.memberships,
            "meta": self.meta,
        }


# =============================================================================
# Private Helpers
# =============================================================================


def _build_backend(ctx: EnhancedHandlerContext) -> DuckDBBackend:
    """Build a DuckDBBackend from context.

    Parameters
    ----------
    ctx
        Enhanced handler context.

    Returns
    -------
    DuckDBBackend
        Constructed backend instance.

    Raises
    ------
    TypeError
        If resolved backend is not DuckDBBackend.
    """
    resource = build_backend_resource(
        ctx.runtime.serving,
        gateway=ctx.gateway,
        options=BackendResourceOptions(graph_runtime=ctx.graph_runtime),
    )

    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        msg = "Expected DuckDBBackend for local_db mode"
        raise TypeError(msg)
    return backend


def _get_int_param(ctx: EnhancedHandlerContext, name: str) -> int | None:
    """Extract optional integer parameter.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    int | None
        Integer value or None.
    """
    value = ctx.params.get(name)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    # Convert string or other types to int
    return int(str(value))


def _get_str_param(ctx: EnhancedHandlerContext, name: str) -> str | None:
    """Extract optional string parameter.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    str | None
        String value or None.
    """
    value = ctx.params.get(name)
    if value is None:
        return None
    return str(value)


def _require_str_param(ctx: EnhancedHandlerContext, name: str) -> str:
    """Extract required string parameter.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    str
        String value.

    Raises
    ------
    ValueError
        If parameter is missing.
    """
    value = ctx.params.get(name)
    if value is None:
        msg = f"{name} parameter is required"
        raise ValueError(msg)
    return str(value)


# =============================================================================
# Handlers
# =============================================================================


def subsystem_list_handler(ctx: EnhancedHandlerContext) -> CliResult[SubsystemListResult]:
    """List inferred subsystems with role/risk metadata.

    Parameters
    ----------
    ctx
        Enhanced handler context with optional params:

        - role: Optional role filter
        - query: Optional search query
        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemListResult]
        Result with subsystem list.
    """
    ctx.logger.debug("Listing subsystems")
    backend = _build_backend(ctx)

    response = backend.list_subsystems(
        limit=_get_int_param(ctx, "limit"),
        role=_get_str_param(ctx, "role"),
        q=_get_str_param(ctx, "query"),
    )

    return CliResult.ok(
        SubsystemListResult(
            subsystems=[row.model_dump() for row in response.subsystems],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_show_handler(ctx: EnhancedHandlerContext) -> CliResult[SubsystemShowResult]:
    """Show subsystem detail and modules.

    Parameters
    ----------
    ctx
        Enhanced handler context with params:

        - subsystem_id: Subsystem ID to show (required)

    Returns
    -------
    CliResult[SubsystemShowResult]
        Result with subsystem details.
    """
    subsystem_id = _require_str_param(ctx, "subsystem_id")
    ctx.logger.debug("Showing subsystem: %s", subsystem_id)

    backend = _build_backend(ctx)
    response = backend.get_subsystem_modules(subsystem_id=subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.debug("Subsystem not found: %s", subsystem_id)
        return CliResult.fail(
            ProblemDetail(
                type="codeintel:subsystem/not-found",
                title="Subsystem not found",
                status=404,
                detail=f"Subsystem not found: {subsystem_id}",
                instance=f"subsystem://{subsystem_id}",
            )
        )

    return CliResult.ok(
        SubsystemShowResult(
            subsystem=response.subsystem.model_dump(),
            modules=[row.model_dump() for row in response.modules],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_profiles_handler(ctx: EnhancedHandlerContext) -> CliResult[SubsystemProfilesResult]:
    """List subsystem profiles from docs.v_subsystem_profile.

    Parameters
    ----------
    ctx
        Enhanced handler context with optional params:

        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemProfilesResult]
        Result with profile list.
    """
    ctx.logger.debug("Listing subsystem profiles")
    backend = _build_backend(ctx)

    response = backend.service.list_subsystem_profiles(limit=_get_int_param(ctx, "limit"))
    profile_response = (
        response
        if isinstance(response, SubsystemProfileResponse)
        else SubsystemProfileResponse.from_domain(response)
    )
    profiles = [
        row if isinstance(row, SubsystemProfileRow) else SubsystemProfileRow.model_validate(row)
        for row in profile_response.profiles
    ]

    return CliResult.ok(
        SubsystemProfilesResult(
            profiles=[row.model_dump() for row in profiles],
            meta=profile_response.meta.model_dump(),
        )
    )


def subsystem_coverage_handler(ctx: EnhancedHandlerContext) -> CliResult[SubsystemCoverageResult]:
    """List subsystem coverage rollups from docs.v_subsystem_coverage.

    Parameters
    ----------
    ctx
        Enhanced handler context with optional params:

        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemCoverageResult]
        Result with coverage list.
    """
    ctx.logger.debug("Listing subsystem coverage")
    backend = _build_backend(ctx)

    response = backend.service.list_subsystem_coverage(limit=_get_int_param(ctx, "limit"))
    coverage_response = (
        response
        if isinstance(response, SubsystemCoverageResponse)
        else SubsystemCoverageResponse.from_domain(response)
    )
    coverage_rows = [
        row if isinstance(row, SubsystemCoverageRow) else SubsystemCoverageRow.model_validate(row)
        for row in coverage_response.coverage
    ]

    return CliResult.ok(
        SubsystemCoverageResult(
            coverage=[row.model_dump() for row in coverage_rows],
            meta=coverage_response.meta.model_dump(),
        )
    )


def subsystem_module_memberships_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[SubsystemMembershipResult]:
    """List subsystem memberships for a module.

    Parameters
    ----------
    ctx
        Enhanced handler context with params:

        - module: Module path to query (required)

    Returns
    -------
    CliResult[SubsystemMembershipResult]
        Result with membership list.
    """
    module = _require_str_param(ctx, "module")
    ctx.logger.debug("Getting subsystem memberships for module: %s", module)

    backend = _build_backend(ctx)
    response = backend.get_module_subsystems(module=module)

    return CliResult.ok(
        SubsystemMembershipResult(
            found=response.found,
            memberships=[row.model_dump() for row in response.memberships],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "SubsystemCoverageResult",
    "SubsystemListResult",
    "SubsystemMembershipResult",
    "SubsystemProfilesResult",
    "SubsystemShowResult",
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
]
