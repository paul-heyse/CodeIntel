"""Hamilton build harness helpers for production-parity tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.providers import Providers, create_default_providers
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.context import SeedPack, TestContext, create_test_context
from tests._helpers.env_options import EnvOptions, GatewayOptions
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.hamilton_manifest_priming import ManifestPriming

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.core.build_manifest import OutputManifest
    from codeintel.core.config.settings import BuildSettings
    from codeintel.storage.gateway import StorageGateway


RepoWriter = Callable[[Path], list[Path]]
RepoStrategy = Literal["canonical", "writer", "none"]


@dataclass(frozen=True)
class BuildEnvSpec:
    """Specification for constructing a BuildEnv in tests."""

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers | None = None
    build_config: BuildConfig | None = None
    settings: BuildSettings | None = None
    profile: str | None = None
    force_targets: frozenset[str] | None = None
    manifest_index: Mapping[str, OutputManifest] | None = None
    validate_outputs: bool = False
    strict_contracts: bool = False


def build_test_env(
    spec: BuildEnvSpec,
) -> BuildEnv:
    """Construct a BuildEnv for unit tests with explicit inputs.

    Parameters
    ----------
    spec
        Build environment specification.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    resolved_providers = spec.providers or create_default_providers(ToolsConfig.default())
    resolved_config = spec.build_config or BuildConfig.empty()
    resolved_settings = spec.settings or TEST_BUILD_SETTINGS
    return BuildEnv(
        gateway=spec.gateway,
        snapshot=spec.snapshot,
        paths=spec.paths,
        providers=resolved_providers,
        config=resolved_config,
        settings=resolved_settings,
        profile=spec.profile,
        force_targets=spec.force_targets or frozenset(),
        manifest_index=spec.manifest_index,
        validate_outputs=spec.validate_outputs,
        strict_contracts=spec.strict_contracts,
    )


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration surface for a Hamilton build harness."""

    repo: str
    commit: str
    profile: str | None = None
    file_backed_db: bool = False
    strict_contracts: bool = False
    validate_outputs: bool = False
    parallel_backend: str = "sequential"
    max_workers: int | None = None
    enable_hamilton_cache: bool = False
    cache_dir: Path | None = None


@dataclass(frozen=True)
class HarnessOpenOptions:
    """Options for constructing a HamiltonBuildHarness."""

    repo_strategy: RepoStrategy = "canonical"
    repo_writer: RepoWriter | None = None
    seed_packs: Sequence[SeedPack] = ()
    gateway_options: GatewayOptions | None = None
    tools_config: ToolsConfig | None = None
    providers: Providers | None = None
    build_config: BuildConfig | None = None


@dataclass
class HamiltonBuildHarness:
    """Production-parity Hamilton execution harness for tests."""

    ctx: TestContext
    env: BuildEnv
    executor: HamiltonBuildExecutor
    config: HarnessConfig
    repo_files: tuple[Path, ...] = ()
    last_result: HamiltonBuildResult | None = None
    _owns_ctx: bool = True

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        harness: HarnessConfig | None = None,
        options: HarnessOpenOptions | None = None,
    ) -> HamiltonBuildHarness:
        """Create an isolated harness bound to a fresh TestContext.

        Returns
        -------
        HamiltonBuildHarness
            New harness wrapping a fresh TestContext.

        Raises
        ------
        ValueError
            If the repo strategy is invalid or required arguments are missing.
        """
        cfg = harness or HarnessConfig(repo="test/repo", commit="deadbeef")
        resolved = options or HarnessOpenOptions()
        repo_root = tmp_path / "repo"
        build_dir = repo_root / "build"
        db_path = build_dir / "db" / "codeintel.duckdb"

        env_opts = EnvOptions(
            repo=cfg.repo,
            commit=cfg.commit,
            file_backed=cfg.file_backed_db,
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path if cfg.file_backed_db else None,
        )
        ctx = create_test_context(
            tmp_path,
            options=env_opts,
            gateway_options=resolved.gateway_options,
        )

        written: list[Path] = []
        if resolved.repo_strategy == "canonical":
            ctx.ensure_canonical_repo()
        elif resolved.repo_strategy == "writer":
            if resolved.repo_writer is None:
                message = "repo_strategy='writer' requires repo_writer"
                raise ValueError(message)
            written = resolved.repo_writer(ctx.repo_root)
        elif resolved.repo_strategy == "none":
            pass
        else:
            message = f"Unknown repo_strategy: {resolved.repo_strategy}"
            raise ValueError(message)

        if resolved.seed_packs:
            ctx.require(*resolved.seed_packs)

        resolved_tools = resolved.tools_config or ToolsConfig.default()
        resolved_providers = resolved.providers or create_default_providers(resolved_tools)
        resolved_build_config = resolved.build_config or load_build_config(ctx.repo_root)

        env = BuildEnv(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
            providers=resolved_providers,
            config=resolved_build_config,
            settings=TEST_BUILD_SETTINGS,
            profile=cfg.profile,
            validate_outputs=cfg.validate_outputs,
            strict_contracts=cfg.strict_contracts,
        )
        executor = HamiltonBuildExecutor(
            profile=cfg.profile,
            parallel_backend=cfg.parallel_backend,
            max_workers=cfg.max_workers,
            enable_cache=cfg.enable_hamilton_cache,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )
        return cls(
            ctx=ctx,
            env=env,
            executor=executor,
            config=cfg,
            repo_files=tuple(written),
            _owns_ctx=True,
        )

    @classmethod
    def wrap(
        cls,
        ctx: TestContext,
        *,
        harness: HarnessConfig | None = None,
        tools_config: ToolsConfig | None = None,
        providers: Providers | None = None,
        build_config: BuildConfig | None = None,
    ) -> HamiltonBuildHarness:
        """Wrap an existing TestContext without owning its lifecycle.

        Returns
        -------
        HamiltonBuildHarness
            Harness bound to the provided TestContext.
        """
        cfg = harness or HarnessConfig(repo=ctx.snapshot.repo, commit=ctx.snapshot.commit)
        resolved_tools = tools_config or ToolsConfig.default()
        resolved_providers = providers or create_default_providers(resolved_tools)
        resolved_build_config = build_config or load_build_config(ctx.repo_root)
        env = BuildEnv(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
            providers=resolved_providers,
            config=resolved_build_config,
            settings=TEST_BUILD_SETTINGS,
            profile=cfg.profile,
            validate_outputs=cfg.validate_outputs,
            strict_contracts=cfg.strict_contracts,
        )
        executor = HamiltonBuildExecutor(
            profile=cfg.profile,
            parallel_backend=cfg.parallel_backend,
            max_workers=cfg.max_workers,
            enable_cache=cfg.enable_hamilton_cache,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )
        return cls(
            ctx=ctx,
            env=env,
            executor=executor,
            config=cfg,
            _owns_ctx=False,
        )

    def close(self) -> None:
        """Close owned context resources."""
        if self._owns_ctx:
            self.ctx.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()

    def with_force_targets(self, *targets: str) -> HamiltonBuildHarness:
        """Set forced targets for subsequent runs.

        Returns
        -------
        HamiltonBuildHarness
            Updated harness instance with forced targets configured.
        """
        self.env = replace(self.env, force_targets=frozenset(targets))
        return self

    def with_profile(self, profile: str | None) -> HamiltonBuildHarness:
        """Override the execution profile.

        Returns
        -------
        HamiltonBuildHarness
            Updated harness instance with the profile set.
        """
        self.env = replace(self.env, profile=profile)
        return self

    def with_build_config(self, config: BuildConfig) -> HamiltonBuildHarness:
        """Override the build configuration.

        Returns
        -------
        HamiltonBuildHarness
            Updated harness instance with the build configuration set.
        """
        self.env = replace(self.env, config=config)
        return self

    def run_targets(self, targets: Iterable[str]) -> HamiltonBuildResult:
        """Execute the Hamilton DAG for provided targets.

        Returns
        -------
        HamiltonBuildResult
            Result bundle produced by the Hamilton executor.
        """
        result = self.executor.run(env=self.env, targets=list(targets))
        self.last_result = result
        return result

    def build_env(self) -> BuildEnv:
        """Return the current BuildEnv for this harness.

        Returns
        -------
        BuildEnv
            Current environment bound to the harness.
        """
        return self.env

    def record(
        self,
        target: str,
        *,
        result: HamiltonBuildResult | None = None,
    ) -> TargetRunRecord:
        """Return the TargetRunRecord for a target from a result.

        Returns
        -------
        TargetRunRecord
            Record for the requested target.

        Raises
        ------
        RuntimeError
            If no result is available or the target record is missing.
        """
        resolved = result or self.last_result
        if resolved is None:
            message = "No HamiltonBuildResult available; call run_targets() first."
            raise RuntimeError(message)
        record = resolved.get_record(target)
        if record is None:
            message = f"No TargetRunRecord found for target {target}"
            raise RuntimeError(message)
        return record

    def require(self, *packs: SeedPack) -> HamiltonBuildHarness:
        """Apply seed packs after construction.

        Returns
        -------
        HamiltonBuildHarness
            Updated harness instance with required seed packs.
        """
        self.ctx.require(*packs)
        return self

    @property
    def artifacts(self) -> HarnessArtifacts:
        """Access artifact helpers rooted at this harness."""
        return HarnessArtifacts(self.ctx.snapshot.repo_root, self.ctx.build_paths)

    @property
    def priming(self) -> ManifestPriming:
        """Access manifest priming helpers for this harness."""
        return ManifestPriming(self)


__all__ = [
    "HamiltonBuildHarness",
    "HarnessConfig",
    "HarnessOpenOptions",
    "RepoStrategy",
    "RepoWriter",
    "build_test_env",
]
