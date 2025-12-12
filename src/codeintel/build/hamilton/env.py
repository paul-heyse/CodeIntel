"""Build environment bundle for Hamilton node execution.

This module defines the BuildEnv dataclass, which is the single frozen input
passed to Hamilton nodes. By consolidating all execution dependencies into
one immutable object, the orchestration interface remains "pure" from
Hamilton's perspective while plugins can access everything they need.

Design Principles
-----------------
1. BuildEnv is frozen/immutable once constructed.
2. All dependencies are injected, enabling testing without mocking.
3. This is passed as an "input" to Hamilton driver execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.config import BuildConfig
    from codeintel.build.providers import Providers
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class BuildEnv:
    """Bundled execution dependencies for Hamilton node execution.

    This is the single input object passed to Hamilton nodes via the driver's
    ``inputs`` parameter. It provides access to all resources needed for
    target execution without requiring global state.

    Attributes
    ----------
    gateway
        Storage gateway for database access and build tracking.
    snapshot
        Repository snapshot reference (repo, commit, root path).
    paths
        Build paths for directory resolution (build_dir, scip_dir, etc.).
    providers
        DI providers for external tools (SCIP indexer, type checker, etc.).
    config
        Build configuration loaded from codeintel.build.toml.
    profile
        Optional policy profile name (e.g., "fast", "full", "default").
        Used to select execution variants in later phases.
    force_targets
        Set of target names that should bypass skip checks and always
        recompute. Used to implement --force CLI flag.

    Examples
    --------
    >>> env = BuildEnv(
    ...     gateway=gateway,
    ...     snapshot=snapshot,
    ...     paths=paths,
    ...     providers=providers,
    ...     config=config,
    ...     profile="default",
    ...     force_targets=frozenset(["function_metrics"]),
    ... )
    >>> driver.execute(["t__risk_factors"], inputs={"env": env})
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    profile: str | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)

    @property
    def repo(self) -> str:
        """Return the repository slug.

        Returns
        -------
        str
            Repository identifier from snapshot.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return the commit SHA.

        Returns
        -------
        str
            Commit identifier from snapshot.
        """
        return self.snapshot.commit

    def is_forced(self, target_name: str) -> bool:
        """Check if a target is in the force set.

        Parameters
        ----------
        target_name
            Target name to check.

        Returns
        -------
        bool
            True if target should bypass skip checks.
        """
        return target_name in self.force_targets


__all__ = [
    "BuildEnv",
]
