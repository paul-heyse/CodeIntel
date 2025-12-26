"""Hamilton-native test execution helpers.

This module provides test execution helpers that use native Hamilton targets
directly. Tests using these helpers execute through the same code paths as
production.

Key Components
--------------
- ``HamiltonTestContext`` - Bundled execution context for Hamilton tests
- ``HamiltonTestBuilder`` - Fluent builder for creating test contexts
- ``execute_hamilton_target`` - Execute a Hamilton target by name

Usage
-----
::

    from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
    from tests._helpers.assertions import assert_target_ok


    def test_modules(tmp_path: Path) -> None:
        with HamiltonBuildHarness.open(tmp_path) as harness:
            result = harness.run_targets(["modules"])
            record = harness.record("modules", result=result)
            assert_target_ok(record)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.providers import create_default_providers
from codeintel.build.settings import DEFAULT_PROFILE_NAME
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from tests._helpers.context import TestContext
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness, HarnessConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.runtime import HamiltonRuntime
    from codeintel.build.providers import Providers
    from codeintel.build.targets import TargetGraph
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class HamiltonTestContext:
    """Bundled execution context for Hamilton tests.

    This class bundles all the components needed to execute Hamilton targets
    in tests. It provides access to the BuildEnv, target graph, and other
    resources needed for execution.

    Attributes
    ----------
    env
        Build environment for Hamilton node execution.
    graph
        Target graph for looking up targets and dependencies.
    runtime
        Hamilton runtime containing the Driver and node mappings.
    harness
        Harness used to execute Hamilton targets.
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    repo_root
        Root path of the test repository.

    Examples
    --------
    >>> ctx = HamiltonTestContext(env=env, graph=graph, ...)
    >>> record = execute_hamilton_target("modules", ctx)
    >>> assert_target_ok(record)
    """

    env: BuildEnv
    graph: TargetGraph
    runtime: HamiltonRuntime
    harness: HamiltonBuildHarness
    gateway: StorageGateway
    snapshot: SnapshotRef
    repo_root: Path


@dataclass
class HamiltonTestBuilder:
    """Fluent builder for Hamilton test execution contexts.

    This builder provides a clean interface for creating test execution
    contexts that use native Hamilton targets. It handles all the wiring
    of providers, paths, and configuration.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    tmp_path
        Temporary directory for test isolation.
    repo_root
        Root path of the repository (defaults to tmp_path/repo).
    build_dir
        Build output directory (defaults to tmp_path/build).
    snapshot_variant
        Snapshot variant for repo/commit identifiers.
    providers
        DI providers (created from default if not specified).
    config
        Build configuration (empty if not specified).
    profile
        Policy profile name.
    force_targets
        Set of targets that bypass skip checks.
    validate_outputs
        Whether to validate outputs against schemas.

    Examples
    --------
    Basic usage:

    >>> builder = HamiltonTestBuilder.create(gateway, tmp_path)
    >>> record = builder.execute_target("modules")

    With custom configuration:

    >>> builder = (
    ...     HamiltonTestBuilder.create(gateway, tmp_path)
    ...     .with_snapshot_variant(DEFAULT_VARIANT)
    ...     .with_force_targets({"modules"})
    ... )
    >>> record = builder.execute_target("modules")
    """

    gateway: StorageGateway
    tmp_path: Path
    repo_root: Path | None = None
    build_dir: Path | None = None
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    providers: Providers | None = None
    config: BuildConfig | None = None
    profile: str = DEFAULT_PROFILE_NAME
    force_targets: frozenset[str] = field(default_factory=frozenset)
    validate_outputs: bool = False
    _runtime: HamiltonRuntime | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        gateway: StorageGateway,
        tmp_path: Path,
        *,
        runtime: HamiltonRuntime | None = None,
    ) -> HamiltonTestBuilder:
        """Create a new builder with required dependencies.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        tmp_path
            Temporary directory for test isolation.
        runtime
            Optional shared Hamilton runtime to reuse across builders.

        Returns
        -------
        HamiltonTestBuilder
            New builder instance.

        Examples
        --------
        >>> builder = HamiltonTestBuilder.create(gateway, tmp_path)
        """
        builder = cls(gateway=gateway, tmp_path=tmp_path)
        if runtime is not None:
            builder.with_runtime(runtime)
        return builder

    def with_repo_root(self, repo_root: Path) -> HamiltonTestBuilder:
        """Set the repository root path.

        Parameters
        ----------
        repo_root
            Root path of the repository.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.repo_root = repo_root
        return self

    def with_build_dir(self, build_dir: Path) -> HamiltonTestBuilder:
        """Set the build output directory.

        Parameters
        ----------
        build_dir
            Build output directory path.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.build_dir = build_dir
        return self

    def with_snapshot_variant(self, variant: SnapshotVariant) -> HamiltonTestBuilder:
        """Set repository snapshot variant.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.snapshot_variant = variant
        return self

    def with_providers(self, providers: Providers) -> HamiltonTestBuilder:
        """Set custom DI providers.

        Parameters
        ----------
        providers
            DI providers container.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.providers = providers
        return self

    def with_config(self, config: BuildConfig) -> HamiltonTestBuilder:
        """Set build configuration.

        Parameters
        ----------
        config
            Build configuration.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.config = config
        return self

    def with_profile(self, profile: str) -> HamiltonTestBuilder:
        """Set the policy profile name.

        Parameters
        ----------
        profile
            Profile name (e.g., "full", "fast", "ci").

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.profile = profile
        return self

    def with_force_targets(self, targets: set[str]) -> HamiltonTestBuilder:
        """Set targets that bypass skip checks.

        Parameters
        ----------
        targets
            Set of target names to force.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.force_targets = frozenset(targets)
        return self

    def with_validation(self, *, enabled: bool = True) -> HamiltonTestBuilder:
        """Enable or disable output validation.

        Parameters
        ----------
        enabled
            Whether to validate outputs against schemas.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self.validate_outputs = enabled
        return self

    def with_runtime(self, runtime: HamiltonRuntime) -> HamiltonTestBuilder:
        """Reuse a shared Hamilton runtime.

        Parameters
        ----------
        runtime
            Hamilton runtime to reuse across tests.

        Returns
        -------
        HamiltonTestBuilder
            Self for method chaining.
        """
        self._runtime = runtime
        return self

    def _resolve_repo_root(self) -> Path:
        """Resolve the repository root path.

        Returns
        -------
        Path
            Repository root path.
        """
        if self.repo_root is not None:
            return self.repo_root
        resolved = self.tmp_path / "repo"
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _resolve_build_dir(self) -> Path:
        """Resolve the build output directory.

        Returns
        -------
        Path
            Build output directory path.
        """
        if self.build_dir is not None:
            return self.build_dir
        resolved = self.tmp_path / "build"
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _resolve_providers(self) -> Providers:
        """Resolve DI providers.

        Returns
        -------
        Providers
            DI providers container.
        """
        if self.providers is not None:
            return self.providers
        tools_config = ToolsConfig.default()
        return create_default_providers(tools_config)

    def _resolve_config(self) -> BuildConfig:
        """Resolve build configuration.

        Returns
        -------
        BuildConfig
            Build configuration.
        """
        if self.config is not None:
            return self.config
        return BuildConfig.empty()

    def _get_runtime(self) -> HamiltonRuntime:
        """Get or create the Hamilton runtime.

        Returns
        -------
        HamiltonRuntime
            Runtime with Driver, target graph, and node mappings.
        """
        if self._runtime is None:
            self._runtime = build_driver(config={"profile": self.profile})
        return self._runtime

    def build_env(self) -> BuildEnv:
        """Build the Hamilton execution environment.

        Returns
        -------
        BuildEnv
            Configured build environment.

        Examples
        --------
        >>> env = builder.build_env()
        >>> driver.execute(["t__modules"], inputs={"env": env})
        """
        return self.build_harness().build_env()

    def build_harness(self) -> HamiltonBuildHarness:
        """Build a Hamilton build harness for execution.

        Returns
        -------
        HamiltonBuildHarness
            Harness configured for Hamilton execution.
        """
        repo_root = self._resolve_repo_root()
        build_dir = self._resolve_build_dir()
        snapshot = self.snapshot_variant.to_snapshot(repo_root=repo_root)
        paths = BuildPaths.from_explicit(build_dir=build_dir)
        ctx = TestContext(snapshot=snapshot, gateway=self.gateway, build_paths=paths)
        harness = HamiltonBuildHarness.wrap(
            ctx,
            harness=HarnessConfig(
                repo=self.snapshot_variant.repo,
                commit=self.snapshot_variant.commit,
                profile=self.profile,
                validate_outputs=self.validate_outputs,
            ),
            providers=self._resolve_providers(),
            build_config=self._resolve_config(),
        )
        if self.force_targets:
            harness.env = replace(harness.env, force_targets=self.force_targets)
        return harness

    def build_context(self) -> HamiltonTestContext:
        """Build a complete test context.

        Returns
        -------
        HamiltonTestContext
            Bundled test execution context.

        Examples
        --------
        >>> ctx = builder.build_context()
        >>> record = execute_hamilton_target("modules", ctx)
        """
        harness = self.build_harness()
        env = harness.build_env()
        runtime = self._get_runtime()
        graph = runtime.graph

        return HamiltonTestContext(
            env=env,
            graph=graph,
            runtime=runtime,
            harness=harness,
            gateway=self.gateway,
            snapshot=env.snapshot,
            repo_root=env.snapshot.repo_root,
        )

    def execute_target(
        self,
        target_name: str,
        *,
        force: bool = False,
        profile: str | None = None,
    ) -> TargetRunRecord:
        """Execute a Hamilton target by name.

        This is a convenience method that builds the context and executes
        the target in one step.

        Parameters
        ----------
        target_name
            Name of the target to execute (e.g., "modules", "goids").
        force
            Whether to bypass skip checks for the target.
        profile
            Optional profile override for this execution.
        force
            Whether to force execution even when skip checks would pass.
        profile
            Optional profile override for this execution.

        Returns
        -------
        TargetRunRecord
            Execution record with status, duration, and row counts.

        Examples
        --------
        >>> record = builder.execute_target("modules")
        >>> assert_target_ok(record)
        >>> print(f"Processed {record.row_counts} rows")
        """
        ctx = self.build_context()
        force_targets = frozenset({target_name}) if force else None
        return execute_hamilton_target(
            target_name,
            ctx,
            force_targets=force_targets,
            profile=profile,
        )

    def execute_targets(
        self,
        targets: Sequence[str],
        *,
        force_targets: frozenset[str] | None = None,
        profile: str | None = None,
    ) -> dict[str, TargetRunRecord]:
        """Execute multiple Hamilton targets in a single driver run.

        Parameters
        ----------
        targets
            Target names to execute.
        force_targets
            Optional target names that bypass skip checks.
        profile
            Optional profile override for this execution.
        force_targets
            Optional target names that bypass skip checks.
        profile
            Optional profile override for this execution.

        Returns
        -------
        dict[str, TargetRunRecord]
            Mapping of target name to TargetRunRecord.
        """
        ctx = self.build_context()
        return execute_hamilton_targets(
            targets,
            ctx,
            force_targets=force_targets,
            profile=profile,
        )

    async def execute_target_async(
        self,
        target_name: str,
        *,
        force: bool = False,
        profile: str | None = None,
    ) -> TargetRunRecord:
        """Execute a Hamilton target asynchronously.

        Parameters
        ----------
        target_name
            Name of the target to execute.
        force
            Whether to bypass skip checks for the target.
        profile
            Optional profile override for this execution.
        force
            Whether to force execution even when skip checks would pass.
        profile
            Optional profile override for this execution.

        Returns
        -------
        TargetRunRecord
            Execution record with status, duration, and row counts.

        Examples
        --------
        >>> record = await builder.execute_target_async("scip")
        >>> assert_target_ok(record)
        """
        ctx = self.build_context()
        force_targets = frozenset({target_name}) if force else None
        return await execute_hamilton_target_async(
            target_name,
            ctx,
            force_targets=force_targets,
            profile=profile,
        )


def execute_hamilton_target(
    target_name: str,
    ctx: HamiltonTestContext,
    *,
    force_targets: frozenset[str] | None = None,
    profile: str | None = None,
) -> TargetRunRecord:
    """Execute a Hamilton target by name.

    This function executes a native Hamilton target via the Hamilton Driver,
    providing the same execution path as production and returning the
    materialized TargetRunRecord.

    Parameters
    ----------
    target_name
        Name of the target to execute (e.g., "modules", "goids").
    ctx
        Hamilton test context with env, graph, and resources.
    force_targets
        Optional target names that bypass skip checks.
    profile
        Optional profile override for this execution.

    Returns
    -------
    TargetRunRecord
        Execution record with status, duration, and row counts.

    Examples
    --------
    >>> ctx = builder.build_context()
    >>> record = execute_hamilton_target("modules", ctx)
    >>> assert_target_ok(record)

    With skip check:

    >>> record = execute_hamilton_target("modules", ctx)
    >>> if record.skipped:
    ...     print("Target was skipped due to matching manifest")
    """
    return asyncio.run(
        execute_hamilton_target_async(
            target_name,
            ctx,
            force_targets=force_targets,
            profile=profile,
        )
    )


def execute_hamilton_targets(
    targets: Iterable[str],
    ctx: HamiltonTestContext,
    *,
    force_targets: frozenset[str] | None = None,
    profile: str | None = None,
) -> dict[str, TargetRunRecord]:
    """Execute multiple Hamilton targets by name.

    Parameters
    ----------
    targets
        Target names to execute.
    ctx
        Hamilton test context with env, graph, and resources.
    force_targets
        Optional target names that bypass skip checks.
    profile
        Optional profile override for this execution.

    Returns
    -------
    dict[str, TargetRunRecord]
        Mapping of target name to execution records.

    Raises
    ------
    RuntimeError
        If a requested target does not return a TargetRunRecord.
    """
    resolved_env = ctx.env
    if force_targets is not None:
        resolved_env = replace(resolved_env, force_targets=force_targets)
    if profile is not None:
        resolved_env = replace(resolved_env, profile=profile)

    result = ctx.harness.executor.run(env=resolved_env, targets=list(targets))
    records: dict[str, TargetRunRecord] = {}
    for target in targets:
        record = result.get_record(target)
        if record is None:
            message = f"No TargetRunRecord found for target {target}"
            raise RuntimeError(message)
        records[target] = record
    return records


async def execute_hamilton_target_async(
    target_name: str,
    ctx: HamiltonTestContext,
    *,
    force_targets: frozenset[str] | None = None,
    profile: str | None = None,
) -> TargetRunRecord:
    """Execute a Hamilton target asynchronously.

    This is the async version of execute_hamilton_target, suitable for
    targets that perform I/O operations or call external tools.

    Parameters
    ----------
    target_name
        Name of the target to execute.
    ctx
        Hamilton test context with env, graph, and resources.
    force_targets
        Optional target names that bypass skip checks.
    profile
        Optional profile override for this execution.

    Returns
    -------
    TargetRunRecord
        Execution record with status, duration, and row counts.
    """
    records = await asyncio.to_thread(
        execute_hamilton_targets,
        [target_name],
        ctx,
        force_targets=force_targets,
        profile=profile,
    )
    return records[target_name]


__all__ = [
    "HamiltonTestBuilder",
    "HamiltonTestContext",
    "execute_hamilton_target",
    "execute_hamilton_target_async",
    "execute_hamilton_targets",
]
