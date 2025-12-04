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

from codeintel.graphs.resources.container import ResourceContainer, ResourceNotFoundError
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.gateway import StorageGateway

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

    assert resource.gateway is graph_gateway
    assert resource.repo_root == tmp_path


def test_storage_resource_resource_name() -> None:
    """StorageResource has correct RESOURCE_NAME."""
    assert StorageResource.RESOURCE_NAME == RESOURCE_NAME


def test_storage_resource_gateway_access(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource provides gateway access."""
    resource = StorageResource(graph_gateway, tmp_path)

    # Gateway should be accessible
    assert resource.gateway is not None
    assert resource.gateway.con is not None


# ---------------------------------------------------------------------------
# Tests: StorageResource with ResourceContainer
# ---------------------------------------------------------------------------


def test_storage_resource_registration(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be registered in container."""
    container = ResourceContainer()
    resource = StorageResource(graph_gateway, tmp_path)

    container.register(resource)

    assert container.has(StorageResource.RESOURCE_NAME)


def test_storage_resource_retrieval(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be retrieved from container."""
    container = ResourceContainer()
    resource = StorageResource(graph_gateway, tmp_path)
    container.register(resource)

    retrieved = typing.cast("StorageResource", container.get(StorageResource.RESOURCE_NAME))

    assert retrieved is resource
    assert retrieved.gateway is graph_gateway


def test_storage_resource_require(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be required from container."""
    container = ResourceContainer()
    resource = StorageResource(graph_gateway, tmp_path)
    container.register(resource)

    required = container.require(StorageResource)

    assert required is resource


def test_storage_resource_not_registered() -> None:
    """Container returns None for unregistered storage."""
    container = ResourceContainer()

    result = container.get(StorageResource.RESOURCE_NAME)

    assert result is None


def test_storage_resource_require_missing_raises() -> None:
    """Require raises KeyError for missing storage resource."""
    container = ResourceContainer()

    with pytest.raises(ResourceNotFoundError):
        container.require(StorageResource)


# ---------------------------------------------------------------------------
# Tests: StorageResource path handling
# ---------------------------------------------------------------------------


def test_storage_resource_path_absolute(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource handles absolute paths."""
    resource = StorageResource(graph_gateway, tmp_path)

    assert resource.repo_root.is_absolute()


def test_storage_resource_path_is_pathlib(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource repo_root is pathlib.Path."""
    resource = StorageResource(graph_gateway, tmp_path)

    assert isinstance(resource.repo_root, Path)


def test_storage_resource_different_paths(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """Multiple StorageResources with different paths."""
    path1 = tmp_path / "repo1"
    path2 = tmp_path / "repo2"
    path1.mkdir()
    path2.mkdir()

    resource1 = StorageResource(graph_gateway, path1)
    resource2 = StorageResource(graph_gateway, path2)

    assert resource1.repo_root != resource2.repo_root
    assert resource1.gateway is resource2.gateway  # Same gateway


# ---------------------------------------------------------------------------
# Tests: StorageResource connection handling
# ---------------------------------------------------------------------------


def test_storage_resource_connection_usable(graph_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource gateway connection is usable."""
    resource = StorageResource(graph_gateway, tmp_path)

    # Should be able to execute a simple query
    result = resource.gateway.con.execute("SELECT 1 AS value").fetchone()

    assert result is not None
    assert result[0] == 1


def test_storage_resource_multiple_resources_same_gateway(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Multiple resources can share same gateway."""
    resource1 = StorageResource(graph_gateway, tmp_path / "a")
    resource2 = StorageResource(graph_gateway, tmp_path / "b")

    # Both should reference same connection
    assert resource1.gateway.con is resource2.gateway.con
