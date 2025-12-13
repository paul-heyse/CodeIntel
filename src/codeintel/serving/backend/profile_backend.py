"""Profile query layer backed by DuckDB repositories.

This module provides the **Query Layer** implementation for profile and
module-related operations.

Layer Hierarchy
---------------
::

    Transport Layer (MCP/HTTP backends: DuckDBBackend, HttpBackend)
         │
         ▼
    Service Layer (LocalQueryService, HttpQueryService)
         │
         ▼
    Query Layer (FunctionQueryLayer, ProfileQueryLayer, etc.) ← This module
         │
         ▼
    Repository Layer (ModuleRepository, SubsystemRepository)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving import domain_models as dm
from codeintel.serving.backend.domain_builders import (
    build_file_hints,
    build_file_profile,
    build_file_summary,
    build_module_architecture,
    build_module_profile,
)
from codeintel.serving.backend.query_api import ProfileQueriesApi
from codeintel.serving.mcp import errors

if TYPE_CHECKING:
    from codeintel.config.graph_helpers import GraphRunScope
    from codeintel.serving.backend.core import BackendContext, DuckDBConnection, DuckDBRepositories
    from codeintel.storage.repositories import ModuleRepository, SubsystemRepository

Message = dm.Message
ResponseMeta = dm.ResponseMeta


@dataclass
class ProfileQueryLayer(ProfileQueriesApi):
    """DuckDB-backed implementation of profile query operations."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def con(self) -> DuckDBConnection:
        """Return the active DuckDB connection."""
        return self.context.gateway.con

    @property
    def modules(self) -> ModuleRepository:
        """Return the lazily constructed module repository."""
        return self.repositories.modules

    @property
    def subsystems(self) -> SubsystemRepository:
        """Return the lazily constructed subsystem repository."""
        return self.repositories.subsystems

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        """
        Return a file profile for the requested path.

        Parameters
        ----------
        rel_path
            Relative file path within the repository.

        Returns
        -------
        dm.FileProfileResult
            File profile payload with metadata.
        """
        row = self.modules.get_file_profile(rel_path)
        if row is None:
            meta = ResponseMeta(
                messages=[
                    Message(
                        code="not_found",
                        severity="warning",
                        detail=f"File profile not found: {rel_path}",
                    )
                ]
            )
            return build_file_profile(None, meta=meta)
        return build_file_profile(row, meta=ResponseMeta())

    def get_file_summary(
        self, *, rel_path: str, scope: GraphRunScope | None = None
    ) -> dm.FileSummaryResult:
        """
        Return a file summary for the requested path.

        Parameters
        ----------
        rel_path
            Relative file path within the repository.
        scope
            Graph scope (unused currently).

        Returns
        -------
        dm.FileSummaryResult
            Summary payload for the file.

        Raises
        ------
        errors.not_found
            If the file summary is missing.
        """
        _ = scope
        row = self.modules.get_file_summary(rel_path)
        if row is None:
            message = f"File summary not found: {rel_path}"
            raise errors.not_found(message)
        return build_file_summary(row, rel_path=rel_path, meta=ResponseMeta())

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        """
        Return a module profile for the requested module.

        Parameters
        ----------
        module
            Module name to retrieve.

        Returns
        -------
        dm.ModuleProfileResult
            Module profile payload.

        Raises
        ------
        errors.not_found
            If the module profile is missing.
        """
        row = self.modules.get_module_profile(module)
        if row is None:
            message = f"Module profile not found: {module}"
            raise errors.not_found(message)
        return build_module_profile(row, meta=ResponseMeta())

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        """
        Return module architecture metrics for the requested module.

        Parameters
        ----------
        module
            Module name to retrieve.

        Returns
        -------
        dm.ModuleArchitectureResult
            Architecture payload for the module.

        Raises
        ------
        errors.not_found
            If the module architecture is missing.
        """
        row = self.modules.get_module_architecture(module)
        if row is None:
            message = f"Module architecture not found: {module}"
            raise errors.not_found(message)
        return build_module_architecture(row, meta=ResponseMeta())

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        """
        Fetch IDE-focused hints for a file by relative path.

        Queries the ``docs.v_ide_hints`` view which aggregates module metrics,
        subsystem context, and risk indicators for IDE consumption.

        Parameters
        ----------
        rel_path
            Relative file path within the repository.

        Returns
        -------
        FileHintsResponse
            Hint rows including subsystem_name, risk levels, and fan metrics.
        """
        hints = self.modules.get_file_hints(rel_path)
        return build_file_hints(hints, rel_path=rel_path, meta=ResponseMeta())


__all__ = ["ProfileQueryLayer"]
