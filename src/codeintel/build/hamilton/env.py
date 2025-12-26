"""Build environment bundle for Hamilton node execution.

This module defines the BuildEnv dataclass, which is the single frozen input
passed to Hamilton nodes. By consolidating all execution dependencies into
one immutable object, the orchestration interface remains "pure" from
Hamilton's perspective while targets can access everything they need.

Design Principles
-----------------
1. BuildEnv is frozen/immutable once constructed.
2. All dependencies are injected, enabling testing without mocking.
3. This is passed as an "input" to Hamilton driver execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.assets.fingerprinting import (
    DEFAULT_FINGERPRINT_POLICY,
)
from codeintel.core.config.settings import BuildSettings, HamiltonExecutionSettings
from codeintel.storage.warehouse import Warehouse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from codeintel.analytics.history.history_timeseries import HistoryTimeseriesOptions
    from codeintel.build.assets.fingerprinting import (
        FingerprintPolicy,
    )
    from codeintel.build.config import BuildConfig
    from codeintel.build.hamilton.run_record_utils import BuildManifestService
    from codeintel.build.providers import Providers
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.core.execution import ExecutionContext, RunContext
    from codeintel.core.registry import RegistryService
    from codeintel.storage import StorageFacade
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
    storage
        Storage facade for non-storage access patterns.
    snapshot
        Repository snapshot reference (repo, commit, root path).
    paths
        Build paths for directory resolution (build_dir, scip_dir, etc.).
    providers
        DI providers for external tools (SCIP indexer, type checker, etc.).
    config
        Build configuration loaded from codeintel.build.toml.
    settings
        Build settings injected by the CLI/runtime boundary.
    execution_settings
        Hamilton execution settings (parallelism + DuckDB options).
    profile
        Optional policy profile name (e.g., "fast", "full", "ci").
        Used to select execution variants in later phases.
    force_targets
        Set of target names that should bypass skip checks and always
        recompute. Used to implement --force CLI flag.
    manifest_index
        Pre-loaded mapping of target names to their manifests for this
        repo/commit. Used to avoid per-target DB round trips during
        skip checks and hash computation.
    validate_outputs
        When True, validate produced datasets against their Pandera schemas
        after write. Validation failures will mark the target as failed and
        block downstream targets.
    history_options
        Optional options for multi-commit history aggregation targets.
    history_db_resolver
        Optional resolver for fetching StorageGateways by commit for
        history aggregation targets.
    fingerprint_policy
        Policy for computing asset version fingerprints. Defaults to STABLE_V1
        for cross-commit reuse capability.
    execution_context
        Optional unified execution context with run metadata, primitives,
        and settings for this build.
    registry
        Optional registry service for dataset and target discovery.

    Examples
    --------
    >>> env = BuildEnv(
    ...     gateway=gateway,
    ...     snapshot=snapshot,
    ...     paths=paths,
    ...     providers=providers,
    ...     config=config,
    ...     profile="full",
    ...     force_targets=frozenset(["function_metrics"]),
    ... )
    >>> driver.execute(["t__risk_factors"], inputs={"env": env})
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    settings: BuildSettings
    storage: StorageFacade | None = None
    execution_settings: HamiltonExecutionSettings = field(default_factory=HamiltonExecutionSettings)
    profile: str | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)
    manifest_index: Mapping[str, OutputManifest] | None = None
    validate_outputs: bool = False
    strict_contracts: bool = False
    history_options: HistoryTimeseriesOptions | None = None
    history_db_resolver: Callable[[str], StorageGateway] | None = None
    fingerprint_policy: FingerprintPolicy = field(
        default_factory=lambda: DEFAULT_FINGERPRINT_POLICY
    )
    execution_context: ExecutionContext | None = None
    registry: RegistryService | None = None

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

    @property
    def run_context(self) -> RunContext | None:
        """Return the unified run context when available."""
        if self.execution_context is None:
            return None
        return self.execution_context.run

    @property
    def warehouse(self) -> Warehouse:
        """Return a storage Warehouse façade for the current gateway.

        Returns
        -------
        Warehouse
            Warehouse wrapper around the build gateway.
        """
        if self.storage is not None:
            return self.storage.warehouse
        return Warehouse(self.gateway)

    @property
    def manifest_service(self) -> BuildManifestService:
        """Return the manifest service for skip evaluation.

        Returns
        -------
        BuildManifestService
            Manifest service used to resolve OutputManifest entries.
        """
        return self.gateway.build


__all__ = [
    "BuildEnv",
]
