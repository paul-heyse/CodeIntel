"""Unified context hierarchy for build operations.

This module defines the base context types for all build operations:

- **ContextPropertiesProtocol**: Common properties shared by all contexts
- **BuildContext**: Base for all build operations (materialization, queries)
- **PathResolver**: Centralized path resolution for build artifacts

Target-specific execution context is provided by `TargetExecutionContext`
in `codeintel.build.context`, which composes `BuildContext` via delegation.

Usage
-----
>>> ctx = BuildContext(gateway=gateway, snapshot=snapshot, paths=paths)
>>> ctx.repo  # Convenient property access
'my-org/my-repo'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.build.session import BuildSession

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.storage.gateway import StorageGateway
from codeintel.storage.helpers.table_key import split_table_key

__all__ = [
    "BuildContext",
    "ContextPropertiesProtocol",
    "PathResolver",
]


@runtime_checkable
class ContextPropertiesProtocol(Protocol):
    """Protocol for common context properties.

    This protocol defines the shared interface for build contexts,
    enabling code that works with any context type.
    """

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        ...

    @property
    def paths(self) -> BuildPaths:
        """Return the build paths."""
        ...

    @property
    def repo(self) -> str:
        """Return the repository slug."""
        ...

    @property
    def commit(self) -> str:
        """Return the commit SHA."""
        ...

    @property
    def repo_root(self) -> Path:
        """Return the repository root path."""
        ...

    @property
    def build_dir(self) -> Path:
        """Return the build directory."""
        ...

    @property
    def scip_dir(self) -> Path:
        """Return the SCIP artifacts directory."""
        ...


@dataclass(frozen=True)
class PathResolver:
    """Centralized path resolution for build artifacts.

    This utility class provides consistent path resolution across
    the build system, eliminating duplicated path formatting logic.

    Attributes
    ----------
    paths
        Build paths configuration.
    snapshot
        Repository snapshot reference.

    Examples
    --------
    >>> resolver = PathResolver(paths=paths, snapshot=snapshot)
    >>> resolver.artifact_path_from_template("{build_dir}/data/{table}.parquet", table="metrics")
    PosixPath('/build/data/metrics.parquet')
    >>> resolver.table_export_path("analytics.function_metrics")
    PosixPath('/export/analytics/function_metrics.parquet')
    """

    paths: BuildPaths
    snapshot: SnapshotRef

    @property
    def build_dir(self) -> Path:
        """Return the build directory.

        Returns
        -------
        Path
            Build output directory.
        """
        return self.paths.build_dir

    @property
    def scip_dir(self) -> Path:
        """Return the SCIP artifacts directory.

        Returns
        -------
        Path
            Directory for SCIP index files.
        """
        return self.paths.scip_dir

    @property
    def export_dir(self) -> Path:
        """Return the export directory.

        Returns
        -------
        Path
            Directory for document output exports.
        """
        return self.paths.document_output_dir

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.snapshot.repo_root

    def artifact_path_from_template(
        self,
        template: str,
        **kwargs: str,
    ) -> Path:
        """Resolve artifact path from a template string.

        Standard template variables:
        - {build_dir}: Build output directory
        - {scip_dir}: SCIP artifacts directory
        - {export_dir}: Export/document output directory
        - {repo_root}: Repository root directory

        Parameters
        ----------
        template
            Path template with {variable} placeholders.
        **kwargs
            Additional template variables.

        Returns
        -------
        Path
            Resolved file path.

        Examples
        --------
        >>> resolver.artifact_path_from_template("{build_dir}/output.json")
        PosixPath('/path/to/build/output.json')
        >>> resolver.artifact_path_from_template("{export_dir}/{name}.parquet", name="data")
        PosixPath('/path/to/export/data.parquet')
        """
        standard_vars = {
            "build_dir": str(self.build_dir),
            "scip_dir": str(self.scip_dir),
            "export_dir": str(self.export_dir),
            "repo_root": str(self.repo_root),
        }
        all_vars = {**standard_vars, **kwargs}
        resolved = template.format(**all_vars)
        return Path(resolved)

    def table_export_path(
        self,
        table_key: str,
        fmt: str = "parquet",
    ) -> Path:
        """Generate export path for a table.

        Parameters
        ----------
        table_key
            Fully-qualified table name (e.g., "analytics.function_metrics").
        fmt
            File format extension (default: "parquet").

        Returns
        -------
        Path
            Export file path.

        Examples
        --------
        >>> resolver.table_export_path("analytics.function_metrics")
        PosixPath('/export/analytics/function_metrics.parquet')
        >>> resolver.table_export_path("core.modules", fmt="jsonl")
        PosixPath('/export/core/modules.jsonl')
        """
        schema, table = split_table_key(table_key)
        return self.export_dir / schema / f"{table}.{fmt}"


@dataclass(frozen=True)
class BuildContext:
    """Base context for all build operations.

    This is the minimal context needed for materialization and basic
    build operations. It provides access to storage, paths, and
    session-scoped caching.

    This class also supports materialization options (validate_schemas,
    owner_target, input_hash) that were previously only in MaterializationContext.
    Both can now be used interchangeably for materialization operations.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference (repo, commit, root path).
    paths
        Build paths for directory resolution.
    session
        Optional build session for caching (created lazily if needed).
    validate_schemas
        When True, validate materialized outputs against Pandera schemas.
    owner_target
        Optional target name that produces assets (for asset catalog).
    input_hash
        Optional input hash from manifest (for asset catalog).

    Examples
    --------
    >>> ctx = BuildContext(gateway=gateway, snapshot=snapshot, paths=paths)
    >>> ctx.repo
    'my-org/my-repo'
    >>> ctx.get_session()  # Returns cached or new session
    BuildSession(...)
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    session: BuildSession | None = field(default=None)
    # Materialization options (enables BuildContext to be used for materialization)
    validate_schemas: bool = field(default=False)
    owner_target: str | None = field(default=None)
    input_hash: str | None = field(default=None)

    @property
    def repo(self) -> str:
        """Return the repository slug.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return the commit SHA.

        Returns
        -------
        str
            Commit identifier.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.snapshot.repo_root

    @property
    def build_dir(self) -> Path:
        """Return the build directory.

        Returns
        -------
        Path
            Build output directory.
        """
        return self.paths.build_dir

    @property
    def scip_dir(self) -> Path:
        """Return the SCIP artifacts directory.

        Returns
        -------
        Path
            Directory for SCIP index files.
        """
        return self.paths.scip_dir

    def get_session(self) -> BuildSession:
        """Get or create a build session for caching.

        If a session was provided at construction, returns it.
        Otherwise, creates a new session.

        Returns
        -------
        BuildSession
            Session for caching hashes and manifests.
        """
        if self.session is not None:
            return self.session
        return BuildSession(snapshot=self.snapshot, gateway=self.gateway)

    @property
    def path_resolver(self) -> PathResolver:
        """Return a path resolver for this context.

        Returns
        -------
        PathResolver
            Utility for resolving artifact paths.

        Examples
        --------
        >>> ctx.path_resolver.table_export_path("analytics.function_metrics")
        PosixPath('/export/analytics/function_metrics.parquet')
        """
        return PathResolver(paths=self.paths, snapshot=self.snapshot)
