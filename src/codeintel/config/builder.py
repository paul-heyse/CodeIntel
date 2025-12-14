"""Unified configuration builder for pipeline setup.

This module provides configuration builders for constructing pipeline contexts.
Step-specific configurations have been migrated to use SnapshotRef + options
dataclasses directly in their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from codeintel.config.primitives import (
    BuildLayoutOptions,
    GraphBackendConfig,
    ScanProfiles,
    ToolBinaries,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.primitives import (
        BuildPaths,
        SnapshotInit,
        SnapshotRef,
    )


@dataclass(frozen=True)
class BuilderDependencies:
    """Optional overrides for builder-scoped dependencies."""

    binaries: ToolBinaries | None = None
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig | None = None

    def resolved(self) -> tuple[ToolBinaries, ScanProfiles | None, GraphBackendConfig]:
        """Return dependency instances with defaults applied.

        Returns
        -------
        tuple[ToolBinaries, ScanProfiles | None, GraphBackendConfig]
            Concrete binaries, scan profiles, and graph backend configuration.
        """
        return (
            self.binaries or ToolBinaries(),
            self.profiles,
            self.graph_backend or GraphBackendConfig(),
        )


@dataclass
class ConfigBuilder:
    """Build pipeline context from a shared snapshot and build paths.

    This builder provides factory methods for constructing pipeline contexts.
    For analytics and graph computations, use SnapshotRef + options dataclasses
    directly instead of step configurations.
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries = field(default_factory=ToolBinaries)
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig = field(default_factory=GraphBackendConfig)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotInit,
        *,
        layout: BuildLayoutOptions | None = None,
        primitives: BuilderDependencies | None = None,
    ) -> Self:
        """Create a builder from snapshot and layout primitives.

        Parameters
        ----------
        snapshot
            Snapshot parameters (repo, commit, repo_root, optional branch).
        layout
            Optional build layout overrides.
        primitives
            Optional dependency overrides (binaries, profiles, graph backend).

        Returns
        -------
        Self
            ConfigBuilder ready to produce pipeline contexts.
        """
        layout_options = layout or BuildLayoutOptions()
        dependencies = primitives or BuilderDependencies()
        snapshot_ref = snapshot.to_snapshot_ref()
        has_layout_overrides = any(
            value is not None
            for value in (
                layout_options.build_dir,
                layout_options.db_path,
                layout_options.document_output_dir,
                layout_options.log_db_path,
            )
        )
        paths = layout_options.materialize(
            snapshot_ref.repo_root,
            check_collisions=has_layout_overrides,
        )
        binaries, profiles, graph_backend = dependencies.resolved()
        profiles = cls._ensure_profiles(profiles)
        return cls(
            snapshot=snapshot_ref,
            paths=paths,
            binaries=binaries,
            profiles=profiles,
            graph_backend=graph_backend,
        )

    @classmethod
    def from_primitives(
        cls,
        snapshot: SnapshotRef,
        paths: BuildPaths,
        *,
        binaries: ToolBinaries | None = None,
        profiles: ScanProfiles | None = None,
        graph_backend: GraphBackendConfig | None = None,
    ) -> Self:
        """Create a builder from pre-constructed primitives.

        Parameters
        ----------
        snapshot
            Snapshot reference for the repository.
        paths
            Build paths configuration.
        binaries
            Optional tool binaries configuration.
        profiles
            Optional scan profiles.
        graph_backend
            Optional graph backend configuration.

        Returns
        -------
        Self
            ConfigBuilder ready to produce pipeline contexts.
        """
        return cls(
            snapshot=snapshot,
            paths=paths,
            binaries=binaries or ToolBinaries(),
            profiles=cls._ensure_profiles(profiles),
            graph_backend=graph_backend or GraphBackendConfig(),
        )

    @staticmethod
    def _ensure_profiles(profiles: ScanProfiles | None) -> ScanProfiles | None:
        """Validate scan profiles and enforce completeness when provided.

        Parameters
        ----------
        profiles
            Optional scan profiles bundle.

        Returns
        -------
        ScanProfiles | None
            Validated profiles or None.

        Raises
        ------
        TypeError
            If provided profiles are not a ScanProfiles instance.
        ValueError
            If provided profiles are missing code or config entries.
        """
        if profiles is None:
            return None
        if not isinstance(profiles, ScanProfiles):
            message = "profiles must be a ScanProfiles instance when provided"
            raise TypeError(message)
        if profiles.code is None or profiles.config is None:
            message = "profiles must include both code and config scan profiles"
            raise ValueError(message)
        return profiles

    def prepare_filesystem(self, *, create_missing_only: bool = True) -> tuple[Path, ...]:
        """Ensure build-related directories exist.

        Parameters
        ----------
        create_missing_only
            When True, create only directories that do not already exist.

        Returns
        -------
        tuple[Path, ...]
            Directories created during preparation.
        """
        targets = (
            self.paths.build_dir,
            self.paths.db_path.parent,
            self.paths.document_output_dir,
            self.paths.scip_dir,
            self.paths.coverage_json.parent,
            self.paths.pytest_report.parent,
            self.paths.tool_cache,
            self.paths.log_db_path.parent,
        )
        created: list[Path] = []
        for target in targets:
            if create_missing_only and target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
        return tuple(created)


__all__ = [
    "BuilderDependencies",
    "ConfigBuilder",
]
