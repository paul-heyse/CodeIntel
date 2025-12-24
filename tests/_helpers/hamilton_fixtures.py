"""Test fixtures for Hamilton build execution tests.

This module provides fixtures that create real Hamilton execution environments
for testing, following the Testing Charter's mandate for production-parity tests.

These fixtures use real storage, real providers, and the same code paths as
production - no monkeypatching, no test-only code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.providers import create_default_providers
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths
from codeintel.core.plugins.execution.profiles import DEFAULT_PROFILE_NAME
from tests._helpers.context import TestContext
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness, HarnessConfig

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class BuildEnvOptions:
    """Optional overrides for building a BuildEnv in tests."""

    build_dir: Path | None = None
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    tools_config: ToolsConfig | None = None
    providers: Providers | None = None
    config: BuildConfig | None = None
    profile: str = DEFAULT_PROFILE_NAME


def make_build_harness(
    *,
    gateway: StorageGateway,
    repo_root: Path,
    options: BuildEnvOptions | None = None,
) -> HamiltonBuildHarness:
    """Create a HamiltonBuildHarness for execution tests.

    This function builds a harness using real providers and paths, so tests can
    reuse a consistent setup when an executor or BuildEnv is needed.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo_root
        Root path of the repository.
    options
        Optional overrides for build paths, snapshot, providers, and profile.

    Returns
    -------
    HamiltonBuildHarness
        Configured build harness.
    """
    resolved_options = options or BuildEnvOptions()
    variant = resolved_options.snapshot_variant
    resolved_build = resolved_options.build_dir or (repo_root / "build")

    snapshot = variant.to_snapshot(repo_root=repo_root)

    paths = BuildPaths.from_explicit(build_dir=resolved_build)

    resolved_tools = resolved_options.tools_config or ToolsConfig.default()
    resolved_providers = resolved_options.providers or create_default_providers(resolved_tools)
    resolved_config = resolved_options.config or BuildConfig.empty()

    ctx = TestContext(snapshot=snapshot, gateway=gateway, build_paths=paths)
    return HamiltonBuildHarness.wrap(
        ctx,
        harness=HarnessConfig(
            repo=variant.repo,
            commit=variant.commit,
            profile=resolved_options.profile,
        ),
        providers=resolved_providers,
        build_config=resolved_config,
    )


def make_build_env(
    *,
    gateway: StorageGateway,
    repo_root: Path,
    options: BuildEnvOptions | None = None,
) -> BuildEnv:
    """Create a BuildEnv for Hamilton execution tests.

    This function creates a real BuildEnv with actual providers and paths,
    suitable for testing Hamilton execution.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo_root
        Root path of the repository.
    options
        Optional overrides for build paths, snapshot, providers, and profile.

    Returns
    -------
    BuildEnv
        Configured build environment.
    """
    harness = make_build_harness(
        gateway=gateway,
        repo_root=repo_root,
        options=options,
    )
    return harness.build_env()


@pytest.fixture
def hamilton_env(
    analytics_gateway: StorageGateway,
    tmp_path: Path,
) -> BuildEnv:
    """Create a BuildEnv for Hamilton tests using TestContext-style setup.

    This fixture provides a complete BuildEnv with real database and
    providers, suitable for testing Hamilton execution.

    Parameters
    ----------
    analytics_gateway
        Storage gateway fixture from conftest.
    tmp_path
        Pytest temp directory fixture.

    Returns
    -------
    BuildEnv
        Configured build environment for tests.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    return make_build_env(
        gateway=analytics_gateway,
        repo_root=repo_root,
        options=BuildEnvOptions(build_dir=build_dir),
    )


@pytest.fixture
def seeded_hamilton_env(
    seeded_analytics_gateway: StorageGateway,
    tmp_path: Path,
) -> BuildEnv:
    """Create a BuildEnv with seeded data for Hamilton tests.

    This fixture uses seeded_analytics_gateway which has basic test data
    pre-populated for integration testing.

    Parameters
    ----------
    seeded_analytics_gateway
        Storage gateway with test data seeded.
    tmp_path
        Pytest temp directory fixture.

    Returns
    -------
    BuildEnv
        Configured build environment with seeded data.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    return make_build_env(
        gateway=seeded_analytics_gateway,
        repo_root=repo_root,
        options=BuildEnvOptions(build_dir=build_dir),
    )


__all__ = [
    "BuildEnvOptions",
    "hamilton_env",
    "make_build_env",
    "make_build_harness",
    "seeded_hamilton_env",
]
