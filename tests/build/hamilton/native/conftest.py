"""Pytest fixtures for native Hamilton migration tests.

This module provides fixtures for the migration test harness including
test repositories and storage gateway access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.build.hamilton.native.harness import (
    MigrationTestHarness,
    create_test_repo_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


@pytest.fixture
def test_repo(tmp_path: Path) -> Path:
    """Create a small test repository with known outputs.

    Returns
    -------
    Path
        Path to the test repository root.
    """
    return create_test_repo_fixture(tmp_path)


@pytest.fixture
def migration_harness(
    gateway: StorageGateway,
) -> MigrationTestHarness:
    """Create a MigrationTestHarness for comparing implementations.

    Parameters
    ----------
    gateway
        Storage gateway from conftest.

    Returns
    -------
    MigrationTestHarness
        Configured test harness.
    """
    return MigrationTestHarness(
        gateway=gateway,
        repo="test/migration-harness",
        commit="test-commit-sha",
    )


# List of native targets that exist and can be tested
NATIVE_ANALYTICS_TARGETS = [
    "external_deps",
    "function_types",
]

NATIVE_INGESTION_TARGETS = [
    "scip",
    "typing",
]

NATIVE_GRAPHS_TARGETS = [
    "call_graph",
    "cfg",
    "dfg",
    "import_graph",
]

NATIVE_EXPORT_TARGETS = [
    "export_jsonl",
    "export_parquet",
]

ALL_NATIVE_TARGETS = (
    NATIVE_ANALYTICS_TARGETS
    + NATIVE_INGESTION_TARGETS
    + NATIVE_GRAPHS_TARGETS
    + NATIVE_EXPORT_TARGETS
)


@pytest.fixture(params=NATIVE_ANALYTICS_TARGETS)
def analytics_target(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for analytics native targets.

    Returns
    -------
    str
        Target name.
    """
    return str(request.param)


@pytest.fixture(params=ALL_NATIVE_TARGETS)
def native_target(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for all native targets.

    Returns
    -------
    str
        Target name.
    """
    return str(request.param)
