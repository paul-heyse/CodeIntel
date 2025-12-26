"""BuildRunContext factory for assembling BuildEnv and execution options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.assets.fingerprinting import DEFAULT_FINGERPRINT_POLICY
from codeintel.build.config import BuildConfig, BuildConfigOverrides, BuildConfigStack
from codeintel.build.execution_policy import ExecutionPolicy
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.config.settings import BuildSettings, HamiltonExecutionSettings
from codeintel.core.execution import ExecutionContext
from codeintel.core.registry import RegistryService
from codeintel.storage import StorageFacade

if TYPE_CHECKING:
    from collections.abc import Mapping as MappingABC

    from codeintel.analytics.history.history_timeseries import HistoryTimeseriesOptions
    from codeintel.build.assets.fingerprinting import FingerprintPolicy
    from codeintel.build.providers import Providers
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class BuildRunContextOverrides:
    """Optional overrides when building a BuildRunContext from an ExecutionContext."""

    execution_options: BuildExecutionOptions | None = None
    config_overrides: BuildConfigOverrides | None = None
    force_targets: frozenset[str] | None = None
    validate_outputs: bool = False
    strict_contracts: bool = False
    manifest_index: MappingABC[str, OutputManifest] | None = None
    fingerprint_policy: FingerprintPolicy | None = None
    history_options: HistoryTimeseriesOptions | None = None
    history_db_resolver: Callable[[str], StorageGateway] | None = None


@dataclass(frozen=True)
class BuildRunContext:
    """Factory for build-time environment and execution options."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    settings: BuildSettings
    config_overrides: BuildConfigOverrides | None = None
    execution_settings: HamiltonExecutionSettings | None = None
    execution_options: BuildExecutionOptions | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)
    validate_outputs: bool = False
    strict_contracts: bool = False
    manifest_index: MappingABC[str, OutputManifest] | None = None
    fingerprint_policy: FingerprintPolicy | None = None
    history_options: HistoryTimeseriesOptions | None = None
    history_db_resolver: Callable[[str], StorageGateway] | None = None
    execution_context: ExecutionContext | None = None

    @staticmethod
    def build_config_stack(
        config: BuildConfig,
        overrides: BuildConfigOverrides | None,
    ) -> BuildConfig:
        """Return the effective config stack for this run.

        Parameters
        ----------
        config
            Base build configuration.
        overrides
            Optional per-target configuration overrides.

        Returns
        -------
        BuildConfig
            Effective configuration stack with overrides applied.
        """
        if overrides is None or overrides.is_empty():
            return config
        return BuildConfigStack.from_base(config, overrides=overrides)

    def build_env(
        self,
        *,
        load_catalogs: bool = True,
        load_schema_service: bool = True,
    ) -> BuildEnv:
        """Construct BuildEnv with merged configuration and options.

        Returns
        -------
        BuildEnv
            Build environment derived from the run context.
        """
        stacked = self.build_config_stack(self.config, self.config_overrides)
        profile = None
        if self.execution_options and self.execution_options.profile:
            profile = self.execution_options.profile
        fingerprint_policy = self.fingerprint_policy or DEFAULT_FINGERPRINT_POLICY
        execution_settings = self.execution_settings or HamiltonExecutionSettings()
        registry_service = None
        if load_catalogs:
            registry_service = RegistryService.from_gateway(
                gateway=self.gateway,
                root=self.snapshot.repo_root,
            )
        if load_schema_service:
            get_schema_service()
        storage_facade = StorageFacade.from_gateway(self.gateway)
        return BuildEnv(
            gateway=self.gateway,
            storage=storage_facade,
            snapshot=self.snapshot,
            paths=self.paths,
            providers=self.providers,
            config=stacked,
            settings=self.settings,
            execution_settings=execution_settings,
            profile=profile,
            force_targets=self.force_targets,
            manifest_index=self.manifest_index,
            validate_outputs=self.validate_outputs,
            strict_contracts=self.strict_contracts,
            history_options=self.history_options,
            history_db_resolver=self.history_db_resolver,
            fingerprint_policy=fingerprint_policy,
            execution_context=self.execution_context,
            registry=registry_service,
        )

    def build_execution_options(self) -> BuildExecutionOptions:
        """Return execution options, defaulting when not configured.

        Returns
        -------
        BuildExecutionOptions
            Execution options for the build run.
        """
        if self.execution_options is not None:
            return self.execution_options
        execution_settings = self.execution_settings or HamiltonExecutionSettings()
        return BuildExecutionOptions(
            profile=None,
            parallel_backend=execution_settings.parallel_backend,
            max_workers=execution_settings.max_workers,
        )

    def execution_policy_for(self, target: OutputTarget) -> ExecutionPolicy:
        """Return resolved execution policy for a target.

        Parameters
        ----------
        target
            Target metadata for which to resolve policy.

        Returns
        -------
        ExecutionPolicy
            Effective execution policy for the target.
        """
        return ExecutionPolicy(
            run_options=self.build_execution_options(),
            target_execution=target.execution,
        )

    @classmethod
    def from_execution_context(
        cls,
        *,
        execution_context: ExecutionContext,
        gateway: StorageGateway,
        providers: Providers,
        config: BuildConfig,
        overrides: BuildRunContextOverrides | None = None,
    ) -> BuildRunContext:
        """Build a BuildRunContext from a unified ExecutionContext.

        Parameters
        ----------
        execution_context
            Unified execution context for the run.
        gateway
            Storage gateway for the target snapshot.
        providers
            Tool providers for the execution run.
        config
            Build configuration for the run.
        overrides
            Optional overrides for run configuration and metadata.

        Returns
        -------
        BuildRunContext
            Materialized run context assembled from the ExecutionContext.
        """
        resolved = overrides or BuildRunContextOverrides()
        return cls(
            snapshot=execution_context.snapshot,
            gateway=gateway,
            paths=execution_context.paths,
            providers=providers,
            config=config,
            config_overrides=resolved.config_overrides,
            settings=execution_context.build_settings,
            execution_settings=execution_context.execution_settings,
            execution_options=resolved.execution_options,
            force_targets=resolved.force_targets or frozenset(),
            validate_outputs=resolved.validate_outputs,
            strict_contracts=resolved.strict_contracts,
            manifest_index=resolved.manifest_index,
            fingerprint_policy=resolved.fingerprint_policy,
            history_options=resolved.history_options,
            history_db_resolver=resolved.history_db_resolver,
            execution_context=execution_context,
        )


__all__ = ["BuildRunContext", "BuildRunContextOverrides"]
