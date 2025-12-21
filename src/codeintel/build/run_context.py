"""BuildRunContext factory for assembling BuildEnv and execution options."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.build.assets.fingerprinting import DEFAULT_FINGERPRINT_POLICY
from codeintel.build.config import BuildConfig, BuildConfigStack
from codeintel.build.execution_policy import ExecutionPolicy
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.run_config import BuildRunConfig
from codeintel.build.target_inventory import get_output_inventory
from codeintel.build.target_metadata import OutputInventory
from codeintel.core.config.settings import BuildSettings, HamiltonExecutionSettings

if TYPE_CHECKING:
    from collections.abc import Mapping as MappingABC

    from codeintel.build.assets.fingerprinting import FingerprintPolicy
    from codeintel.build.providers import Providers
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class BuildRunContext:
    """Factory for build-time environment and execution options."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    settings: BuildSettings
    execution_settings: HamiltonExecutionSettings | None = None
    run_config: BuildRunConfig | None = None
    execution_options: BuildExecutionOptions | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)
    validate_outputs: bool = False
    strict_contracts: bool = False
    manifest_index: MappingABC[str, OutputManifest] | None = None
    output_inventory: OutputInventory | None = None
    fingerprint_policy: FingerprintPolicy | None = None

    @staticmethod
    def build_config_stack(
        config: BuildConfig,
        run_config: BuildRunConfig | None,
    ) -> BuildConfig:
        """Return the effective config stack for this run.

        Parameters
        ----------
        config
            Base build configuration.
        run_config
            Optional run configuration with per-target overrides.

        Returns
        -------
        BuildConfig
            Effective configuration stack with run overrides applied.
        """
        overrides = None
        if run_config is not None:
            overrides = run_config.config_overrides_for_target
        run_overrides: Mapping[str, Mapping[str, object]] | None = None
        if overrides is not None:
            run_overrides = _RunOverridesView(overrides)
        return BuildConfigStack.from_base(config, run_overrides=run_overrides)

    def build_env(self) -> BuildEnv:
        """Construct BuildEnv with merged configuration and options.

        Returns
        -------
        BuildEnv
            Build environment derived from the run context.
        """
        stacked = self.build_config_stack(self.config, self.run_config)
        profile = None
        if self.execution_options and self.execution_options.profile:
            profile = self.execution_options.profile
        elif self.run_config is not None:
            profile = self.run_config.profile_name
        fingerprint_policy = self.fingerprint_policy or DEFAULT_FINGERPRINT_POLICY
        output_inventory = self.output_inventory
        if output_inventory is None:
            output_inventory = get_output_inventory()
        execution_settings = self.execution_settings or HamiltonExecutionSettings()
        return BuildEnv(
            gateway=self.gateway,
            snapshot=self.snapshot,
            paths=self.paths,
            providers=self.providers,
            config=stacked,
            settings=self.settings,
            execution_settings=execution_settings,
            profile=profile,
            force_targets=self.force_targets,
            manifest_index=self.manifest_index,
            output_inventory=output_inventory,
            validate_outputs=self.validate_outputs,
            strict_contracts=self.strict_contracts,
            fingerprint_policy=fingerprint_policy,
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
        profile = self.run_config.profile_name if self.run_config is not None else None
        execution_settings = self.execution_settings or HamiltonExecutionSettings()
        return BuildExecutionOptions(
            profile=profile,
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


class _RunOverridesView(Mapping[str, Mapping[str, object]]):
    """Lazy view over BuildRunConfig overrides."""

    def __init__(self, overrides_fn: Callable[[str], Mapping[str, Any]]) -> None:
        self._overrides_fn = overrides_fn
        self._cache: dict[str, Mapping[str, object]] = {}

    def __getitem__(self, key: str) -> Mapping[str, object]:
        """Return overrides for the requested target name.

        Parameters
        ----------
        key
            Target name.

        Returns
        -------
        Mapping[str, object]
            Override mapping for the target.
        """
        if key in self._cache:
            return self._cache[key]
        value = self._overrides_fn(key)
        self._cache[key] = value
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate cached override keys.

        Returns
        -------
        Iterator[str]
            Iterator over cached override keys.
        """
        return iter(self._cache)

    def __len__(self) -> int:
        """Return count of cached override entries.

        Returns
        -------
        int
            Number of cached override entries.
        """
        return len(self._cache)


__all__ = ["BuildRunContext"]
