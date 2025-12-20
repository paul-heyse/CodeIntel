"""Test fixtures for Hamilton build execution tests.

This module provides fixtures that create real Hamilton execution environments
for testing, following the Testing Charter's mandate for production-parity tests.

These fixtures use real storage, real providers, and the same code paths as
production - no monkeypatching, no test-only code paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton import BuildEnv
from codeintel.build.providers import create_default_providers
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.plugins.execution.profiles import DEFAULT_PROFILE_NAME
from tests._helpers.build import TEST_BUILD_SETTINGS

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


def make_build_env(
    *,
    gateway: StorageGateway,
    repo_root: Path,
    build_dir: Path | None = None,
    snapshot_info: tuple[str, str] = ("test/repo", "abc123"),
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
    build_dir
        Build output directory (defaults to repo_root/build).
    snapshot_info
        Tuple of (repo_slug, commit_sha) for snapshot reference.

    Returns
    -------
    BuildEnv
        Configured build environment.
    """
    repo, commit = snapshot_info
    resolved_build = build_dir or (repo_root / "build")

    snapshot = SnapshotRef(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
    )

    paths = BuildPaths.from_explicit(build_dir=resolved_build)

    tools_config = ToolsConfig.default()

    providers = create_default_providers(tools_config)

    config = BuildConfig.empty()

    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        settings=TEST_BUILD_SETTINGS,
        profile=DEFAULT_PROFILE_NAME,
    )


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
        build_dir=build_dir,
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
        build_dir=build_dir,
    )


__all__ = [
    "hamilton_env",
    "make_build_env",
    "seeded_hamilton_env",
]
