"""Hamilton build harness helpers for production-parity tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.hamilton.graph_validation import validate_graph
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.providers import Providers, create_default_providers
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.context import SeedPack, TestContext, create_test_context
from tests._helpers.env_options import EnvOptions, GatewayOptions
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.hamilton_manifest_priming import ManifestPriming
from tests._helpers.scenarios import ScenarioConfig
from tests._helpers.tooling_audit import require_tooling

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


def _prepare_harness_context(
    tmp_path: Path,
    *,
    config: HarnessConfig,
    options: HarnessOpenOptions,
    scenario: ScenarioConfig,
) -> tuple[TestContext, list[Path], Providers, BuildConfig]:
    """Build a TestContext and supporting config for a harness.

    Returns
    -------
    tuple[TestContext, list[Path], Providers, BuildConfig]
        Context, written repo files, providers, and build config.

    Raises
    ------
    ValueError
        If the repo strategy is invalid or required arguments are missing.
    """
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    resolved_variant = options.snapshot_variant or scenario.snapshot_variant or DEFAULT_VARIANT

    env_opts = EnvOptions(
        file_backed=config.file_backed_db or scenario.file_backed,
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path if config.file_backed_db else None,
        snapshot_variant=resolved_variant,
    )
    ctx = create_test_context(
        tmp_path,
        options=env_opts,
        gateway_options=options.gateway_options,
    )
    if scenario.extra:
        ctx.extra.update(scenario.extra)

    written: list[Path] = []
    if options.repo_strategy == "canonical":
        ctx.ensure_canonical_repo()
    elif options.repo_strategy == "writer":
        if options.repo_writer is None:
            message = "repo_strategy='writer' requires repo_writer"
            raise ValueError(message)
        written = options.repo_writer(ctx.repo_root)
    elif options.repo_strategy != "none":
        message = f"Unknown repo_strategy: {options.repo_strategy}"
        raise ValueError(message)

    seed_packs = tuple(scenario.seed_packs) + tuple(options.seed_packs)
    if seed_packs:
        ctx.require(*seed_packs)
    if scenario.write_files and options.repo_strategy == "none":
        ctx.ensure_canonical_repo()

    resolved_tools = options.tools_config or ToolsConfig.default()
    require_tooling(resolved_tools, repo_root=ctx.repo_root)
    resolved_providers = options.providers or create_default_providers(resolved_tools)
    resolved_build_config = options.build_config or load_build_config(ctx.repo_root)
    return ctx, written, resolved_providers, resolved_build_config


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration surface for a Hamilton build harness."""

    repo: str
    commit: str
    profile: str | None = None
    file_backed_db: bool = True
    strict_contracts: bool = True
    validate_outputs: bool = True
    parallel_backend: str = "threadpool"
    max_workers: int | None = 4
    enable_hamilton_cache: bool = True
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
    snapshot_variant: SnapshotVariant | None = None
    scenario: ScenarioConfig | None = None

    @classmethod
    def production_repo(
        cls,
        *,
        repo_strategy: RepoStrategy = "canonical",
        seed_packs: Sequence[SeedPack] = (),
        snapshot_variant: SnapshotVariant | None = None,
    ) -> HarnessOpenOptions:
        """Return production-parity open options for repo-backed tests.

        Parameters
        ----------
        repo_strategy
            Repo strategy to use (canonical or writer).
        seed_packs
            Optional seed packs to apply.
        snapshot_variant
            Snapshot variant to apply.

        Returns
        -------
        HarnessOpenOptions
            Production-parity open options.
        """
        return cls(
            repo_strategy=repo_strategy,
            seed_packs=seed_packs,
            snapshot_variant=snapshot_variant,
        )


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

        """
        cfg = harness or HarnessConfig(repo="test/repo", commit="deadbeef")
        if cfg.enable_hamilton_cache and cfg.cache_dir is None:
            cfg = replace(cfg, cache_dir=tmp_path / "hamilton_cache")
        resolved = options or HarnessOpenOptions()
        scenario = resolved.scenario or ScenarioConfig()
        ctx, written, resolved_providers, resolved_build_config = _prepare_harness_context(
            tmp_path,
            config=cfg,
            options=resolved,
            scenario=scenario,
        )

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
        if cfg.enable_hamilton_cache and cfg.cache_dir is None:
            cfg = replace(cfg, cache_dir=ctx.build_paths.build_dir / "hamilton_cache")
        resolved_tools = tools_config or ToolsConfig.default()
        require_tooling(resolved_tools, repo_root=ctx.repo_root)
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

    @staticmethod
    def validate_graph() -> None:
        """Validate Hamilton DAG invariants for the build graph.

        Raises
        ------
        AssertionError
            If graph validation reports errors or warnings.
        """
        result = validate_graph()
        if result.errors or result.warnings:
            message = (
                f"Graph validation failed: errors={len(result.errors)} "
                f"warnings={len(result.warnings)}"
            )
            raise AssertionError(message)

    @staticmethod
    def assert_output_inventory_consistent() -> None:
        """Assert DAG-derived output inventory matches declared contracts.

        Raises
        ------
        AssertionError
            If the resolved output inventory diverges from declared contracts.
        """
        service = get_target_metadata_service()
        catalog = service.system.catalog
        issues: list[str] = []

        for target in catalog.all_targets:
            expected_tables = tuple(sorted(target.contract.table_keys))
            expected_artifacts = tuple(sorted(target.contract.artifact_names))
            expected_templates = {
                artifact.name: artifact.path_template for artifact in target.contract.artifacts
            }

            table_outputs = catalog.table_outputs_by_target.get(target.name, ())
            artifact_outputs = catalog.artifact_outputs_by_target.get(target.name, ())

            observed_tables = tuple(sorted(output.key for output in table_outputs))
            observed_artifacts = tuple(sorted(output.key for output in artifact_outputs))
            observed_templates = {
                output.key: output.artifact_path_template for output in artifact_outputs
            }

            if expected_tables != observed_tables:
                issues.append(
                    "Target contract table_keys differ from DAG outputs "
                    f"for {target.name}: expected={expected_tables} observed={observed_tables}"
                )
            if expected_artifacts != observed_artifacts:
                issues.append(
                    "Target contract artifact_names differ from DAG outputs "
                    f"for {target.name}: expected={expected_artifacts} observed={observed_artifacts}"
                )
            if expected_templates != observed_templates:
                issues.append(
                    "Target contract artifact templates differ from DAG outputs "
                    f"for {target.name}: expected={expected_templates} observed={observed_templates}"
                )

        if issues:
            message = "Output inventory mismatch:\n" + "\n".join(f"- {issue}" for issue in issues)
            raise AssertionError(message)

    def assert_incremental_behavior(
        self,
        target: str,
        *,
        touch_path: Path | None = None,
        mutate_repo: Callable[[Path], None] | None = None,
    ) -> None:
        """Assert target skip behavior across identical and modified inputs.

        Parameters
        ----------
        target
            Target name to execute.
        touch_path
            Optional path to touch between runs.
        mutate_repo
            Optional callback to mutate repo state between runs.

        Raises
        ------
        AssertionError
            If expected skip or rerun behavior is not observed.
        """
        first = self.run_targets([target])
        first_record = self.record(target, result=first)
        if not first_record.success:
            message = f"Expected first run of {target} to succeed"
            raise AssertionError(message)

        second = self.run_targets([target])
        second_record = self.record(target, result=second)
        if not second_record.skipped:
            message = f"Expected second run of {target} to be skipped"
            raise AssertionError(message)

        if mutate_repo is not None:
            mutate_repo(self.ctx.repo_root)
        elif touch_path is not None:
            touch_path.parent.mkdir(parents=True, exist_ok=True)
            touch_path.touch()

        third = self.run_targets([target])
        third_record = self.record(target, result=third)
        if not third_record.success:
            message = f"Expected third run of {target} to succeed after mutation"
            raise AssertionError(message)

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
            message = _format_missing_record_error(target, resolved)
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


def _format_missing_record_error(target: str, result: HamiltonBuildResult) -> str:
    parts = [f"No TargetRunRecord found for target {target}."]
    if result.error:
        parts.append(f"build_error={result.error}")
    if result.failed_targets:
        parts.append(f"failed_targets={result.failed_targets}")
    if result.skipped_targets:
        parts.append(f"skipped_targets={result.skipped_targets}")
    if result.computed_targets:
        parts.append(f"computed_targets={result.computed_targets}")
    return " ".join(parts)


__all__ = [
    "HamiltonBuildHarness",
    "HarnessConfig",
    "HarnessOpenOptions",
    "RepoStrategy",
    "RepoWriter",
    "build_test_env",
]
