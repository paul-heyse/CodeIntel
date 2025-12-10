"""Tests for graph resources and dependency injection.

This module tests the resource registry, protocols, and storage resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, cast

import pytest

from codeintel.core.resources import ResourceNotFoundError, ResourceProviderBase, ResourceRegistry
from codeintel.graphs.resources import ResourceProvider
from codeintel.graphs.resources.storage import StorageResource
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_RESOURCE_NAME: Final[str] = "test_resource"
FACTORY_RESOURCE_NAME: Final[str] = "factory_resource"
STORAGE_RESOURCE_NAME: Final[str] = "storage"
TEST_VALUE: Final[str] = "test_value"
FACTORY_VALUE: Final[str] = "factory_value"
EXPECTED_ONE: Final[int] = 1
EXPECTED_TWO: Final[int] = 2
EXPECTED_THREE: Final[int] = 3
EXPECTED_FORTY_TWO: Final[int] = 42


# ---------------------------------------------------------------------------
# Test Resource Implementations
# ---------------------------------------------------------------------------


@dataclass
class _TestResourceProvider:
    """Simple test resource provider."""

    RESOURCE_NAME: ClassVar[str] = TEST_RESOURCE_NAME

    value: str = TEST_VALUE
    invalidate_count: int = 0

    @property
    def resource_name(self) -> str:
        """Return resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> str:
        """Return test value.

        Returns
        -------
        str
            Test value.
        """
        return self.value

    def invalidate(self) -> None:
        """Track invalidation calls."""
        self.invalidate_count += 1


@dataclass
class _SecondTestResourceProvider:
    """Second test resource provider with different name."""

    RESOURCE_NAME: ClassVar[str] = "test_resource_2"

    value: str = TEST_VALUE
    invalidate_count: int = 0

    @property
    def resource_name(self) -> str:
        """Return resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> str:
        """Return test value.

        Returns
        -------
        str
            Test value.
        """
        return self.value

    def invalidate(self) -> None:
        """Track invalidation calls."""
        self.invalidate_count += 1


@dataclass
class _FactoryResourceProvider:
    """Factory-created resource provider."""

    RESOURCE_NAME: ClassVar[str] = FACTORY_RESOURCE_NAME

    _value: str = FACTORY_VALUE

    @property
    def resource_name(self) -> str:
        """Return resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> str:
        """Return factory value.

        Returns
        -------
        str
            Factory value.
        """
        return self._value

    def invalidate(self) -> None:
        """No-op invalidation."""


class _ConcreteBaseProvider(ResourceProviderBase[str]):
    """Concrete implementation of ResourceProviderBase for testing."""

    RESOURCE_NAME: ClassVar[str] = "concrete_base_provider"

    def __init__(self, name: str, value: str) -> None:
        """Initialize the provider.

        Parameters
        ----------
        name
            Resource name (stored as instance attribute for test compatibility).
        value
            Resource value to return.
        """
        super().__init__()
        self._name = name
        self._value = value

    @property
    def resource_name(self) -> str:
        """Return the resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self._name

    def _load(self) -> str:
        """Load the resource value.

        Returns
        -------
        str
            The resource value.
        """
        return self._value


# ===========================================================================
# ResourceRegistry Tests
# ===========================================================================


def test_registry_register_and_require() -> None:
    """Registry registers and retrieves providers."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    registry.register_provider(provider)
    result = registry.require_by_name(TEST_RESOURCE_NAME)

    expect_equal(result, TEST_VALUE)


def test_registry_require_by_name() -> None:
    """Registry retrieves resource by name."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    registry.register_provider(provider)
    result = registry.require_by_name(TEST_RESOURCE_NAME)

    expect_equal(result, TEST_VALUE)


def test_registry_has_registered() -> None:
    """Registry reports registered resource."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    expect_false(registry.has_by_name(TEST_RESOURCE_NAME))
    registry.register_provider(provider)
    expect_true(registry.has_by_name(TEST_RESOURCE_NAME))


def test_registry_has_factory() -> None:
    """Registry reports factory-registered resource."""
    registry = ResourceRegistry()

    def factory() -> _FactoryResourceProvider:
        return _FactoryResourceProvider()

    expect_false(registry.has_by_name(FACTORY_RESOURCE_NAME))
    registry.register_factory(FACTORY_RESOURCE_NAME, factory)
    expect_true(registry.has_by_name(FACTORY_RESOURCE_NAME))


def test_registry_require_not_found() -> None:
    """Registry raises error for missing resource."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError):
        registry.require_by_name(TEST_RESOURCE_NAME)


def test_registry_require_by_name_not_found() -> None:
    """Registry raises error for missing named resource."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError):
        registry.require_by_name("nonexistent")


def test_registry_factory_lazy_creation() -> None:
    """Registry creates resource from factory on first access."""
    registry = ResourceRegistry()
    creation_count = 0

    def factory() -> _FactoryResourceProvider:
        nonlocal creation_count
        creation_count += 1
        return _FactoryResourceProvider()

    registry.register_factory(FACTORY_RESOURCE_NAME, factory)

    # Factory not called yet
    expect_equal(creation_count, 0)

    # First access creates resource
    registry.require_by_name(FACTORY_RESOURCE_NAME)
    expect_equal(creation_count, EXPECTED_ONE)

    # Second access uses cached
    registry.require_by_name(FACTORY_RESOURCE_NAME)
    expect_equal(creation_count, EXPECTED_ONE)


def test_registry_invalidate_all() -> None:
    """Registry invalidates all resources."""
    registry = ResourceRegistry()
    provider1 = _TestResourceProvider()
    provider2 = _SecondTestResourceProvider()

    registry.register_provider(provider1)
    registry.register_provider(provider2)

    registry.invalidate()

    expect_equal(provider1.invalidate_count, EXPECTED_ONE)
    expect_equal(provider2.invalidate_count, EXPECTED_ONE)


def test_registry_cleanup() -> None:
    """Registry cleanup invalidates and clears."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    registry.register_provider(provider)
    registry.cleanup()

    expect_false(registry.has_by_name(TEST_RESOURCE_NAME))
    expect_equal(provider.invalidate_count, EXPECTED_ONE)


def test_registry_registered_names() -> None:
    """Registry returns all registered names."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    def factory() -> _FactoryResourceProvider:
        return _FactoryResourceProvider()

    registry.register_provider(provider)
    registry.register_factory(FACTORY_RESOURCE_NAME, factory)

    names = registry.registered_names

    expect_in(TEST_RESOURCE_NAME, names)
    expect_in(FACTORY_RESOURCE_NAME, names)
    expect_equal(len(names), EXPECTED_TWO)


def test_registry_overwrite_allows_new_value() -> None:
    """Registry allows overwriting provider with new value."""
    registry = ResourceRegistry()
    provider1 = _TestResourceProvider()
    provider2 = _TestResourceProvider(value="new_value")

    registry.register_provider(provider1)
    # Second registration overwrites (with warning)
    registry.register_provider(provider2)

    # The new value should be stored
    result = registry.require_by_name(TEST_RESOURCE_NAME)
    expect_equal(result, "new_value")


# ===========================================================================
# ResourceProviderBase Tests
# ===========================================================================


class _CountingProvider(ResourceProviderBase[str]):
    """Provider that counts load calls."""

    RESOURCE_NAME: ClassVar[str] = "counting_provider"
    load_count: ClassVar[int] = 0
    _return_value: str = "value"

    def __init__(self, name: str) -> None:
        """Initialize the provider.

        Parameters
        ----------
        name
            Resource name (stored as instance attribute for test compatibility).
        """
        super().__init__()
        self._name = name

    @property
    def resource_name(self) -> str:
        """Return the resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self._name

    def _load(self) -> str:
        """Load and count.

        Returns
        -------
        str
            Loaded value.
        """
        _CountingProvider.load_count += 1
        return self._return_value

    @classmethod
    def reset_count(cls) -> None:
        """Reset the load counter."""
        cls.load_count = 0


def test_base_provider_lazy_loading() -> None:
    """ResourceProviderBase loads resource lazily on first get."""
    _CountingProvider.reset_count()
    provider = _CountingProvider("test")

    # Not loaded yet - load_count is 0
    expect_equal(_CountingProvider.load_count, 0)

    # First access loads
    result = provider.get()
    expect_equal(result, "value")
    expect_equal(_CountingProvider.load_count, EXPECTED_ONE)


def test_base_provider_invalidate() -> None:
    """ResourceProviderBase invalidate allows re-loading."""
    _CountingProvider.reset_count()
    provider = _CountingProvider("test")

    # Load and cache
    provider.get()
    expect_equal(_CountingProvider.load_count, EXPECTED_ONE)

    # Second get uses cache
    provider.get()
    expect_equal(_CountingProvider.load_count, EXPECTED_ONE)

    # Invalidate clears, next get reloads
    provider.invalidate()
    provider.get()
    expect_equal(_CountingProvider.load_count, EXPECTED_TWO)


def test_base_provider_reloads_after_invalidate() -> None:
    """ResourceProviderBase reloads after invalidation."""
    _CountingProvider.reset_count()
    provider = _CountingProvider("test")

    provider.get()
    expect_equal(_CountingProvider.load_count, EXPECTED_ONE)

    provider.invalidate()
    provider.get()
    expect_equal(_CountingProvider.load_count, EXPECTED_TWO)


def test_base_provider_resource_name() -> None:
    """ResourceProviderBase returns resource name."""
    provider = _ConcreteBaseProvider("my_resource", "value")

    expect_equal(provider.resource_name, "my_resource")


def test_base_provider_load_not_implemented() -> None:
    """ResourceProviderBase._load must be overridden."""

    class UnimplementedProvider(ResourceProviderBase[str]):
        RESOURCE_NAME: ClassVar[str] = "unimplemented"

    provider = UnimplementedProvider()

    with pytest.raises(NotImplementedError):
        provider.get()


# ===========================================================================
# ResourceProvider Protocol Tests
# ===========================================================================


def test_resource_provider_protocol_conformance() -> None:
    """Test resources conform to ResourceProvider protocol."""
    provider = _TestResourceProvider()

    expect_is_instance(provider, ResourceProvider)
    expect_true(hasattr(provider, "resource_name"))
    expect_true(hasattr(provider, "get"))
    expect_true(hasattr(provider, "invalidate"))


def test_resource_provider_base_protocol_conformance() -> None:
    """ResourceProviderBase conforms to ResourceProvider protocol."""
    provider = _ConcreteBaseProvider("test", "value")

    expect_is_instance(provider, ResourceProvider)


# ---------------------------------------------------------------------------
# StorageResource fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_resource(graph_gateway: StorageGateway, tmp_path: Path) -> StorageResource:
    """Provide a reusable StorageResource instance for graph tests.

    Returns
    -------
    StorageResource
        Storage resource bound to the graph gateway and tmp_path.
    """
    return StorageResource(gateway=graph_gateway, _repo_root=tmp_path)


@pytest.fixture
def storage_registry(storage_resource: StorageResource) -> ResourceRegistry:
    """Provide a registry pre-loaded with StorageResource.

    Returns
    -------
    ResourceRegistry
        Registry containing the storage resource provider.
    """
    registry = ResourceRegistry()
    registry.register_provider(storage_resource)
    return registry


# ===========================================================================
# StorageResource Tests
# ===========================================================================


def test_storage_resource_creation(storage_resource: StorageResource, tmp_path: Path) -> None:
    """StorageResource can be created."""
    expect_equal(storage_resource.resource_name, STORAGE_RESOURCE_NAME)
    expect_equal(storage_resource.repo_root, tmp_path)
    expect_true(storage_resource.gateway is not None)


def test_storage_resource_resource_name_constant() -> None:
    """StorageResource exposes RESOURCE_NAME constant."""
    expect_equal(StorageResource.RESOURCE_NAME, STORAGE_RESOURCE_NAME)


def test_storage_resource_get_returns_self(storage_resource: StorageResource) -> None:
    """StorageResource.get returns self."""
    result = storage_resource.get()

    expect_true(result is storage_resource)


def test_storage_resource_invalidate_noop(storage_resource: StorageResource) -> None:
    """StorageResource.invalidate is a no-op."""
    storage_resource.invalidate()


def test_storage_resource_gateway_access(storage_resource: StorageResource) -> None:
    """StorageResource exposes gateway and connection."""
    expect_true(storage_resource.gateway.con is not None)


def test_storage_resource_path_absolute(storage_resource: StorageResource) -> None:
    """StorageResource repo_root is absolute path."""
    expect_true(storage_resource.repo_root.is_absolute())


def test_storage_resource_path_is_pathlib(storage_resource: StorageResource) -> None:
    """StorageResource repo_root is pathlib.Path."""
    expect_is_instance(storage_resource.repo_root, Path)


def test_storage_resource_read_source(storage_resource: StorageResource, tmp_path: Path) -> None:
    """StorageResource reads source files."""
    test_file = tmp_path / "test.py"
    test_content = "print('hello')"
    test_file.write_text(test_content)

    result = storage_resource.read_source("test.py")

    expect_equal(result, test_content)


def test_storage_resource_read_source_not_found(
    storage_resource: StorageResource,
) -> None:
    """StorageResource returns None for missing files."""
    result = storage_resource.read_source("nonexistent.py")

    expect_true(result is None)


def test_storage_resource_execute_query(storage_resource: StorageResource) -> None:
    """StorageResource executes queries."""
    result = storage_resource.execute_query("SELECT 1 as value")

    expect_equal(len(result.rows), EXPECTED_ONE)


def test_storage_resource_execute_query_with_params(storage_resource: StorageResource) -> None:
    """StorageResource executes queries with parameters."""
    result = storage_resource.execute_query("SELECT ? + ? as value", [1, 2])

    expect_equal(len(result.rows), EXPECTED_ONE)
    expect_equal(result.rows[0][0], EXPECTED_THREE)


def test_storage_resource_execute_query_empty_result(storage_resource: StorageResource) -> None:
    """StorageResource handles queries with empty results."""
    storage_resource.gateway.con.execute("CREATE TEMP TABLE test_empty (id INT)")
    result = storage_resource.execute_query("SELECT * FROM test_empty WHERE id > 999")

    expect_equal(len(result.rows), 0)


def test_storage_resource_execute_mutation(storage_resource: StorageResource) -> None:
    """StorageResource executes mutations."""
    storage_resource.gateway.con.execute("CREATE TEMP TABLE test_mut (id INT, name VARCHAR)")
    mutation_sql = "INSERT INTO test_mut VALUES (1, 'test') RETURNING id"
    result = storage_resource.execute_mutation(mutation_sql)

    expect_equal(result, EXPECTED_ONE)


def test_storage_resource_execute_mutation_with_params(
    storage_resource: StorageResource,
) -> None:
    """StorageResource executes mutations with parameters."""
    storage_resource.gateway.con.execute("CREATE TEMP TABLE test_mut2 (id INT, name VARCHAR)")
    result = storage_resource.execute_mutation(
        "INSERT INTO test_mut2 VALUES (?, ?) RETURNING id", [42, "test"]
    )

    expect_equal(result, EXPECTED_FORTY_TWO)


def test_storage_resource_execute_mutation_multiple_rows(
    storage_resource: StorageResource,
) -> None:
    """StorageResource handles multi-row mutations."""
    storage_resource.gateway.con.execute("CREATE TEMP TABLE test_mut3 (id INT, name VARCHAR)")
    storage_resource.gateway.con.execute(
        "INSERT INTO test_mut3 VALUES (1, 'a'), (2, 'b'), (3, 'c')"
    )

    update_sql = "UPDATE test_mut3 SET name = 'updated' WHERE id > 0 RETURNING id"
    result = storage_resource.execute_mutation(update_sql)

    expect_equal(result, EXPECTED_ONE)


def test_storage_resource_registration(storage_registry: ResourceRegistry) -> None:
    """StorageResource can be registered in registry."""
    expect_true(storage_registry.has_by_name(StorageResource.RESOURCE_NAME))


def test_storage_resource_retrieval(storage_registry: ResourceRegistry) -> None:
    """StorageResource can be retrieved from registry."""
    retrieved = storage_registry.get_by_name(StorageResource.RESOURCE_NAME)

    expect_true(retrieved is not None)
    if retrieved is None:
        return

    typed_retrieved = cast(StorageResource, retrieved)
    expect_true(typed_retrieved.resource_name == STORAGE_RESOURCE_NAME)


def test_storage_resource_require(storage_registry: ResourceRegistry) -> None:
    """StorageResource can be required from registry."""
    required = storage_registry.require_by_name(StorageResource.RESOURCE_NAME)

    expect_true(required is not None)


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


def test_storage_resource_multiple_resources_same_gateway(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Multiple resources can share same gateway."""
    path1 = tmp_path / "repo1"
    path2 = tmp_path / "repo2"
    path1.mkdir()
    path2.mkdir()

    resource1 = StorageResource(graph_gateway, path1)
    resource2 = StorageResource(graph_gateway, path2)

    expect_true(resource1.gateway.con is resource2.gateway.con)
    expect_true(resource1.repo_root != resource2.repo_root)


def test_storage_resource_connection_usable(storage_resource: StorageResource) -> None:
    """StorageResource gateway connection is usable."""
    result = storage_resource.gateway.con.execute("SELECT 1 AS value").fetchone()

    expect_true(result is not None)
    if result is not None:
        expect_equal(result[0], EXPECTED_ONE)


def test_storage_resource_protocol_conformance(storage_resource: StorageResource) -> None:
    """StorageResource conforms to ResourceProvider protocol."""
    expect_is_instance(storage_resource, ResourceProvider)


# ===========================================================================
# ResourceNotFoundError Tests
# ===========================================================================


def test_resource_not_found_error_message() -> None:
    """ResourceNotFoundError includes resource name."""
    error = ResourceNotFoundError("my_resource")

    expect_in("my_resource", str(error))
    expect_equal(error.resource_name, "my_resource")


def test_resource_not_found_error_is_exception() -> None:
    """ResourceNotFoundError is an Exception."""
    error = ResourceNotFoundError("test")

    expect_is_instance(error, Exception)
