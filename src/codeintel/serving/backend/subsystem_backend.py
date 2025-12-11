"""Subsystem query layer backed by DuckDB repositories.

This module provides the **Query Layer** implementation for subsystem-related
operations.

Layer Hierarchy
---------------
::

    Transport Layer (MCP/HTTP backends: DuckDBBackend, HttpBackend)
         │
         ▼
    Service Layer (LocalQueryService, HttpQueryService)
         │
         ▼
    Query Layer (SubsystemQueryLayer, etc.) ← This module
         │
         ▼
    Repository Layer (SubsystemRepository)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving import domain_models as dm
from codeintel.serving.backend.domain_builders import (
    build_module_subsystems,
    build_subsystem_coverage,
    build_subsystem_modules,
    build_subsystem_profile,
    build_subsystem_search,
    build_subsystem_summary,
)
from codeintel.serving.backend.pagination import clamp_limit
from codeintel.serving.backend.query_api import SubsystemQueriesApi

if TYPE_CHECKING:
    from codeintel.serving.backend.core import BackendContext, DuckDBRepositories
    from codeintel.storage.repositories import SubsystemRepository

ResponseMeta = dm.ResponseMeta


@dataclass
class SubsystemQueryLayer(SubsystemQueriesApi):
    """DuckDB-backed implementation of subsystem query operations."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def subsystems(self) -> SubsystemRepository:
        """Return the lazily constructed subsystem repository."""
        return self.repositories.subsystems

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSummaryResult:
        """
        List subsystem summaries with limit clamping.

        Returns
        -------
        dm.SubsystemSummaryResult
            Subsystem summaries plus pagination metadata.
        """
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystems(
            limit=limit_clamp.limit_or_default(self.context.limits.default_limit),
            role=role,
            query=q,
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        return build_subsystem_summary(rows, meta=meta)

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        """
        Return subsystem memberships for a module.

        Returns
        -------
        dm.ModuleSubsystemResult
            Subsystem memberships for the module.
        """
        rows = self.subsystems.list_subsystems_for_module(module)
        return build_module_subsystems(rows, meta=ResponseMeta())

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """
        Fetch subsystem detail and member modules by subsystem identifier.

        Queries the ``docs.v_subsystem_summary`` view for subsystem metadata
        and ``docs.v_subsystem_modules`` for membership rows. This follows the
        repository pattern consistent with other architecture endpoints.

        Parameters
        ----------
        subsystem_id
            Unique subsystem identifier.
        module_limit
            Maximum modules to return (defaults to backend limit).

        Returns
        -------
        SubsystemModulesResponse
            Subsystem summary and module membership list.
        """
        limit_clamp = clamp_limit(
            module_limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        subsystem_row = self.subsystems.get_subsystem_summary(subsystem_id)
        meta = ResponseMeta(
            requested_limit=module_limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        if subsystem_row is None:
            return build_subsystem_modules(None, [], meta=meta)
        rows = self.subsystems.list_subsystem_modules(subsystem_id)[: limit_clamp.applied]
        return build_subsystem_modules(subsystem_row, rows, meta=meta)

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSearchResult:
        """
        Search subsystems with limit clamping.

        Returns
        -------
        dm.SubsystemSearchResult
            Search results plus pagination metadata.
        """
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.search_subsystems(
            limit=limit_clamp.limit_or_default(self.context.limits.default_limit),
            role=role,
            query=q,
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        return build_subsystem_search(rows, meta=meta)

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """
        Summarize a subsystem and include module memberships.

        Returns
        -------
        dm.SubsystemModulesResult
            Subsystem summary with module memberships.
        """
        return self.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=module_limit)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        """
        List subsystem profiles with limit clamping.

        Returns
        -------
        dm.SubsystemProfileResult
            Subsystem profiles plus pagination metadata.
        """
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystem_profiles(
            limit=limit_clamp.limit_or_default(self.context.limits.default_limit)
        )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=limit_clamp.messages,
        )
        return build_subsystem_profile(rows, meta=meta)

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        """
        List subsystem coverage rollups with limit clamping.

        Returns
        -------
        dm.SubsystemCoverageResult
            Coverage rollups plus pagination metadata.
        """
        limit_clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystem_coverage(
            limit=limit_clamp.limit_or_default(self.context.limits.default_limit)
        )
        messages = list(limit_clamp.messages)
        if not rows:
            messages.append(
                dm.Message(
                    code="not_found",
                    severity="warning",
                    detail="No subsystem coverage found",
                )
            )
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            messages=messages,
        )
        return build_subsystem_coverage(rows, meta=meta)


__all__ = ["SubsystemQueryLayer"]
