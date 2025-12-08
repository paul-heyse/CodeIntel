"""Extended tests for graph resources storage module.

This module provides additional test coverage for the storage resource
from `codeintel.graphs.resources.storage`, including:

- StorageResource initialization and access
- Gateway injection
- Path management
- Resource registration patterns
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Final

import pytest

from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_is_instance, expect_true

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOURCE_NAME: Final = "storage"


# ---------------------------------------------------------------------------
# Tests: StorageResource initialization
# ---------------------------------------------------------------------------


def test_storage_resource_init(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource initializes with gateway and path."""
    resource = StorageResource(graph_gateway, tmp_path)

    expect_true(resource.gateway is graph_gateway)
    expect_equal(resource.repo_root, tmp_path)


def test_storage_resource_resource_name() -> None:
    """StorageResource has correct RESOURCE_NAME."""
    expect_equal(StorageResource.RESOURCE_NAME, RESOURCE_NAME)


def test_storage_resource_gateway_access(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource provides gateway access."""
    resource = StorageResource(graph_gateway, tmp_path)

    # Gateway should be accessible
    expect_true(resource.gateway is not None)
    if resource.gateway is not None:
        expect_true(resource.gateway.con is not None)


# ---------------------------------------------------------------------------
# Tests: StorageResource with ResourceRegistry
# ---------------------------------------------------------------------------


def test_storage_resource_registration(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be registered in registry."""
    registry = ResourceRegistry()
    resource = StorageResource(graph_gateway, tmp_path)

    registry.register_provider(resource)

    expect_true(registry.has_by_name(StorageResource.RESOURCE_NAME))


def test_storage_resource_retrieval(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be retrieved from registry."""
    registry = ResourceRegistry()
    resource = StorageResource(graph_gateway, tmp_path)
    registry.register_provider(resource)

    retrieved = typing.cast("StorageResource", registry.get_by_name(StorageResource.RESOURCE_NAME))

    expect_true(retrieved is resource)
    expect_true(retrieved.gateway is graph_gateway)


def test_storage_resource_require(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be required from registry."""
    registry = ResourceRegistry()
    resource = StorageResource(graph_gateway, tmp_path)
    registry.register_provider(resource)

    required = registry.require_by_name(StorageResource.RESOURCE_NAME)

    expect_true(required is resource)


def test_storage_resource_not_registered() -> None:
    """Registry raises for unregistered storage."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError):
        registry.get_by_name(StorageResource.RESOURCE_NAME)


def test_storage_resource_require_missing_raises() -> None:
    """Require raises KeyError for missing storage resource."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError):
        registry.require_by_name(StorageResource.RESOURCE_NAME)


# ---------------------------------------------------------------------------
# Tests: StorageResource path handling
# ---------------------------------------------------------------------------


def test_storage_resource_path_absolute(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource handles absolute paths."""
    resource = StorageResource(graph_gateway, tmp_path)

    expect_true(resource.repo_root.is_absolute())


def test_storage_resource_path_is_pathlib(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource repo_root is pathlib.Path."""
    resource = StorageResource(graph_gateway, tmp_path)

    expect_is_instance(resource.repo_root, Path)


def test_storage_resource_different_paths(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """Multiple StorageResources with different paths."""
    path1 = tmp_path / "repo1"
    path2 = tmp_path / "repo2"
    path1.mkdir()
    path2.mkdir()

    resource1 = StorageResource(graph_gateway, path1)
    resource2 = StorageResource(graph_gateway, path2)

    expect_true(resource1.repo_root != resource2.repo_root)
    expect_true(resource1.gateway is resource2.gateway)  # Same gateway


# ---------------------------------------------------------------------------
# Tests: StorageResource connection handling
# ---------------------------------------------------------------------------


def test_storage_resource_connection_usable(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource gateway connection is usable."""
    resource = StorageResource(graph_gateway, tmp_path)

    # Should be able to execute a simple query
    result = resource.gateway.con.execute("SELECT 1 AS value").fetchone()

    expect_true(result is not None)
    if result is not None:
        expect_equal(result[0], 1)


def test_storage_resource_multiple_resources_same_gateway(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Multiple resources can share same gateway."""
    resource1 = StorageResource(graph_gateway, tmp_path / "a")
    resource2 = StorageResource(graph_gateway, tmp_path / "b")

    # Both should reference same connection
    expect_true(resource1.gateway.con is resource2.gateway.con)
