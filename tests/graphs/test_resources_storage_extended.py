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
from typing import TYPE_CHECKING, Final

import pytest

from codeintel.graphs.resources.container import ResourceContainer, ResourceNotFoundError
from codeintel.graphs.resources.storage import StorageResource
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOURCE_NAME: Final = "storage"


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_gateway() -> StorageGateway:
    """Create a gateway for storage tests.

    Returns
    -------
    StorageGateway
        Configured gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    return gateway


# ---------------------------------------------------------------------------
# Tests: StorageResource initialization
# ---------------------------------------------------------------------------


def test_storage_resource_init(tmp_path: Path) -> None:
    """StorageResource initializes with gateway and path."""
    gateway = _make_gateway()
    try:
        resource = StorageResource(gateway, tmp_path)

        assert resource.gateway is gateway
        assert resource.repo_root == tmp_path
    finally:
        gateway.close()


def test_storage_resource_resource_name() -> None:
    """StorageResource has correct RESOURCE_NAME."""
    assert StorageResource.RESOURCE_NAME == RESOURCE_NAME


def test_storage_resource_gateway_access(tmp_path: Path) -> None:
    """StorageResource provides gateway access."""
    gateway = _make_gateway()
    try:
        resource = StorageResource(gateway, tmp_path)

        # Gateway should be accessible
        assert resource.gateway is not None
        assert resource.gateway.con is not None
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: StorageResource with ResourceContainer
# ---------------------------------------------------------------------------


def test_storage_resource_registration(tmp_path: Path) -> None:
    """StorageResource can be registered in container."""
    gateway = _make_gateway()
    try:
        container = ResourceContainer()
        resource = StorageResource(gateway, tmp_path)

        container.register(resource)

        assert container.has(StorageResource.RESOURCE_NAME)
    finally:
        gateway.close()


def test_storage_resource_retrieval(tmp_path: Path) -> None:
    """StorageResource can be retrieved from container."""
    gateway = _make_gateway()
    try:
        container = ResourceContainer()
        resource = StorageResource(gateway, tmp_path)
        container.register(resource)

        retrieved = typing.cast("StorageResource", container.get(StorageResource.RESOURCE_NAME))

        assert retrieved is resource
        assert retrieved.gateway is gateway
    finally:
        gateway.close()


def test_storage_resource_require(tmp_path: Path) -> None:
    """StorageResource can be required from container."""
    gateway = _make_gateway()
    try:
        container = ResourceContainer()
        resource = StorageResource(gateway, tmp_path)
        container.register(resource)

        required = container.require(StorageResource)

        assert required is resource
    finally:
        gateway.close()


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


def test_storage_resource_path_absolute(tmp_path: Path) -> None:
    """StorageResource handles absolute paths."""
    gateway = _make_gateway()
    try:
        resource = StorageResource(gateway, tmp_path)

        assert resource.repo_root.is_absolute()
    finally:
        gateway.close()


def test_storage_resource_path_is_pathlib(tmp_path: Path) -> None:
    """StorageResource repo_root is pathlib.Path."""
    gateway = _make_gateway()
    try:
        resource = StorageResource(gateway, tmp_path)

        assert isinstance(resource.repo_root, Path)
    finally:
        gateway.close()


def test_storage_resource_different_paths(tmp_path: Path) -> None:
    """Multiple StorageResources with different paths."""
    gateway = _make_gateway()
    try:
        path1 = tmp_path / "repo1"
        path2 = tmp_path / "repo2"
        path1.mkdir()
        path2.mkdir()

        resource1 = StorageResource(gateway, path1)
        resource2 = StorageResource(gateway, path2)

        assert resource1.repo_root != resource2.repo_root
        assert resource1.gateway is resource2.gateway  # Same gateway
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: StorageResource connection handling
# ---------------------------------------------------------------------------


def test_storage_resource_connection_usable(tmp_path: Path) -> None:
    """StorageResource gateway connection is usable."""
    gateway = _make_gateway()
    try:
        resource = StorageResource(gateway, tmp_path)

        # Should be able to execute a simple query
        result = resource.gateway.con.execute("SELECT 1 AS value").fetchone()

        assert result is not None
        assert result[0] == 1
    finally:
        gateway.close()


def test_storage_resource_multiple_resources_same_gateway(tmp_path: Path) -> None:
    """Multiple resources can share same gateway."""
    gateway = _make_gateway()
    try:
        resource1 = StorageResource(gateway, tmp_path / "a")
        resource2 = StorageResource(gateway, tmp_path / "b")

        # Both should reference same connection
        assert resource1.gateway.con is resource2.gateway.con
    finally:
        gateway.close()
