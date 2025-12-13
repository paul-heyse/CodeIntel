"""Unified context hierarchy for build operations.

This module defines the base context types for all build operations,
creating a coherent hierarchy that replaces overlapping context types:

- **ContextPropertiesMixin**: Common properties shared by all contexts
- **BuildContext**: Base for all build operations (materialization, queries)
- **ExecutionContext**: Extended for target plugin execution

The hierarchy enables code reuse while preserving the specific needs
of different build phases:

```
ContextPropertiesMixin (shared properties)
    │
    ├── BuildContext (base)
    │       ├── gateway, snapshot, paths, session
    │       └── materialization options
    │
    └── ExecutionContext (target-specific)
            ├── target, contract, parameters
            └── resources
```

Usage
-----
>>> ctx = BuildContext(gateway=gateway, snapshot=snapshot, paths=paths)
>>> ctx.repo  # Convenient property access
'my-org/my-repo'

>>> exec_ctx = ExecutionContext(
...     gateway=gateway,
...     snapshot=snapshot,
...     paths=paths,
...     target=my_target,
... )
>>> exec_ctx.target_name
'my_target'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.build.session import BuildSession

if TYPE_CHECKING:
    from codeintel.build.contracts import OutputContract
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "BuildContext",
    "ContextPropertiesProtocol",
    "ExecutionContext",
    "PathResolver",
]


@runtime_checkable
class ContextPropertiesProtocol(Protocol):
    """Protocol for common context properties.

    This protocol defines the shared interface between BuildContext and
    ExecutionContext, enabling code that works with either context type.
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
        schema, table = table_key.split(".", 1)
        return self.export_dir / schema / f"{table}.{fmt}"


@dataclass(frozen=True)
class BuildContext:
    """Base context for all build operations.

    This is the minimal context needed for materialization and basic
    build operations. It provides access to storage, paths, and
    session-scoped caching.

    Extended by ExecutionContext for target plugin execution.

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


@dataclass
class ExecutionContext:
    """Extended context for target plugin execution.

    Adds target-specific information and resources needed by plugin
    execute() methods. This is the recommended base for plugin contexts.

    Unlike BuildContext, this is mutable to allow subclasses like
    TargetExecutionContext to track write operations.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths for directory resolution.
    target
        The OutputTarget being executed.
    parameters
        Tuning parameters for this target.
    session
        Optional build session for caching.
    validate_schemas
        When True, validate materialized outputs against Pandera schemas.
    input_hash
        Optional input hash from manifest (for asset catalog).

    Examples
    --------
    >>> exec_ctx = ExecutionContext(
    ...     gateway=gateway,
    ...     snapshot=snapshot,
    ...     paths=paths,
    ...     target=my_target,
    ... )
    >>> exec_ctx.target_name
    'my_target'
    >>> exec_ctx.contract.table_keys
    ('analytics.function_metrics',)
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    target: OutputTarget
    parameters: TargetParameters | None = field(default=None)
    session: BuildSession | None = field(default=None)
    # Materialization options
    validate_schemas: bool = field(default=False)
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

    @property
    def contract(self) -> OutputContract:
        """Return the target's output contract.

        Returns
        -------
        OutputContract
            Tables and artifacts this target produces.
        """
        return self.target.contract

    @property
    def target_name(self) -> str:
        """Return the target name.

        Returns
        -------
        str
            Target identifier.
        """
        return self.target.name

    def get_session(self) -> BuildSession:
        """Get or create a build session for caching.

        Returns
        -------
        BuildSession
            Session for caching hashes and manifests.
        """
        if self.session is not None:
            return self.session
        return BuildSession(snapshot=self.snapshot, gateway=self.gateway)

    def artifact_path(self, artifact_name: str) -> Path:
        """Resolve an artifact path from the contract.

        Parameters
        ----------
        artifact_name
            Name of the artifact in the contract.

        Returns
        -------
        Path
            Resolved file path.

        Raises
        ------
        KeyError
            If artifact is not in the contract.
        """
        spec = self.contract.get_artifact(artifact_name)
        if spec is None:
            available = ", ".join(self.contract.artifact_names)
            msg = f"Artifact '{artifact_name}' not in contract. Available: {available}"
            raise KeyError(msg)

        template = spec.path_template
        resolved = template.format(
            build_dir=self.build_dir,
            scip_dir=self.scip_dir,
            export_dir=self.paths.document_output_dir,
            repo_root=self.repo_root,
        )
        return Path(resolved)

    @property
    def owner_target(self) -> str | None:
        """Return the target name as owner for asset tracking.

        Returns
        -------
        str | None
            Target name for asset catalog ownership.
        """
        return self.target.name

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

    def to_build_context(self) -> BuildContext:
        """Create a BuildContext from this execution context.

        Returns
        -------
        BuildContext
            Immutable build context with same gateway/snapshot/paths.
        """
        return BuildContext(
            gateway=self.gateway,
            snapshot=self.snapshot,
            paths=self.paths,
            session=self.session,
            validate_schemas=self.validate_schemas,
            owner_target=self.target.name,
            input_hash=self.input_hash,
        )
