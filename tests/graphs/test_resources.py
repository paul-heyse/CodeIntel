"""Tests for graph resources and dependency injection.

This module tests the resource registry, protocols, and storage resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import pytest

from codeintel.core.resources import ResourceNotFoundError, ResourceProviderBase, ResourceRegistry
from codeintel.graphs.resources import ResourceProvider
from codeintel.graphs.resources.storage import StorageResource

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

    assert result == TEST_VALUE


def test_registry_require_by_name() -> None:
    """Registry retrieves resource by name."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    registry.register_provider(provider)
    result = registry.require_by_name(TEST_RESOURCE_NAME)

    assert result == TEST_VALUE


def test_registry_has_registered() -> None:
    """Registry reports registered resource."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    assert registry.has_by_name(TEST_RESOURCE_NAME) is False
    registry.register_provider(provider)
    assert registry.has_by_name(TEST_RESOURCE_NAME) is True


def test_registry_has_factory() -> None:
    """Registry reports factory-registered resource."""
    registry = ResourceRegistry()

    def factory() -> _FactoryResourceProvider:
        return _FactoryResourceProvider()

    assert registry.has_by_name(FACTORY_RESOURCE_NAME) is False
    registry.register_factory(FACTORY_RESOURCE_NAME, factory)
    assert registry.has_by_name(FACTORY_RESOURCE_NAME) is True


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
    assert creation_count == 0

    # First access creates resource
    registry.require_by_name(FACTORY_RESOURCE_NAME)
    assert creation_count == EXPECTED_ONE

    # Second access uses cached
    registry.require_by_name(FACTORY_RESOURCE_NAME)
    assert creation_count == EXPECTED_ONE


def test_registry_invalidate_all() -> None:
    """Registry invalidates all resources."""
    registry = ResourceRegistry()
    provider1 = _TestResourceProvider()
    provider2 = _SecondTestResourceProvider()

    registry.register_provider(provider1)
    registry.register_provider(provider2)

    registry.invalidate()

    assert provider1.invalidate_count == EXPECTED_ONE
    assert provider2.invalidate_count == EXPECTED_ONE


def test_registry_cleanup() -> None:
    """Registry cleanup invalidates and clears."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    registry.register_provider(provider)
    registry.cleanup()

    assert registry.has_by_name(TEST_RESOURCE_NAME) is False
    assert provider.invalidate_count == EXPECTED_ONE


def test_registry_registered_names() -> None:
    """Registry returns all registered names."""
    registry = ResourceRegistry()
    provider = _TestResourceProvider()

    def factory() -> _FactoryResourceProvider:
        return _FactoryResourceProvider()

    registry.register_provider(provider)
    registry.register_factory(FACTORY_RESOURCE_NAME, factory)

    names = registry.registered_names

    assert TEST_RESOURCE_NAME in names
    assert FACTORY_RESOURCE_NAME in names
    assert len(names) == EXPECTED_TWO


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
    assert result == "new_value"


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
    assert _CountingProvider.load_count == 0

    # First access loads
    result = provider.get()
    assert result == "value"
    assert _CountingProvider.load_count == EXPECTED_ONE


def test_base_provider_invalidate() -> None:
    """ResourceProviderBase invalidate allows re-loading."""
    _CountingProvider.reset_count()
    provider = _CountingProvider("test")

    # Load and cache
    provider.get()
    assert _CountingProvider.load_count == EXPECTED_ONE

    # Second get uses cache
    provider.get()
    assert _CountingProvider.load_count == EXPECTED_ONE

    # Invalidate clears, next get reloads
    provider.invalidate()
    provider.get()
    assert _CountingProvider.load_count == EXPECTED_TWO


def test_base_provider_reloads_after_invalidate() -> None:
    """ResourceProviderBase reloads after invalidation."""
    _CountingProvider.reset_count()
    provider = _CountingProvider("test")

    provider.get()
    assert _CountingProvider.load_count == EXPECTED_ONE

    provider.invalidate()
    provider.get()
    assert _CountingProvider.load_count == EXPECTED_TWO


def test_base_provider_resource_name() -> None:
    """ResourceProviderBase returns resource name."""
    provider = _ConcreteBaseProvider("my_resource", "value")

    assert provider.resource_name == "my_resource"


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

    assert isinstance(provider, ResourceProvider)
    assert hasattr(provider, "resource_name")
    assert hasattr(provider, "get")
    assert hasattr(provider, "invalidate")


def test_resource_provider_base_protocol_conformance() -> None:
    """ResourceProviderBase conforms to ResourceProvider protocol."""
    provider = _ConcreteBaseProvider("test", "value")

    assert isinstance(provider, ResourceProvider)


# ===========================================================================
# StorageResource Tests
# ===========================================================================


def test_storage_resource_creation(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource can be created."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    assert resource.resource_name == STORAGE_RESOURCE_NAME
    assert resource.repo_root == tmp_path


def test_storage_resource_get_returns_self(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource.get returns self."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    result = resource.get()

    assert result is resource


def test_storage_resource_invalidate_noop(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource.invalidate is a no-op."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    # Should not raise
    resource.invalidate()


def test_storage_resource_read_source(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource reads source files."""
    # Create a test file
    test_file = tmp_path / "test.py"
    test_content = "print('hello')"
    test_file.write_text(test_content)

    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    result = resource.read_source("test.py")

    assert result == test_content


def test_storage_resource_read_source_not_found(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource returns None for missing files."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    result = resource.read_source("nonexistent.py")

    assert result is None


def test_storage_resource_execute_query(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource executes queries."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    result = resource.execute_query("SELECT 1 as value")

    assert len(result.rows) == EXPECTED_ONE


def test_storage_resource_execute_query_with_params(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource executes queries with parameters."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    result = resource.execute_query("SELECT ? + ? as value", [1, 2])

    assert len(result.rows) == EXPECTED_ONE
    assert result.rows[0][0] == EXPECTED_THREE


def test_storage_resource_execute_query_empty_result(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource handles queries with empty results."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)
    # Query that returns no rows
    fresh_gateway.con.execute("CREATE TEMP TABLE test_empty (id INT)")
    result = resource.execute_query("SELECT * FROM test_empty WHERE id > 999")

    assert len(result.rows) == 0


def test_storage_resource_execute_mutation(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """StorageResource executes mutations."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    # Create a temp table and insert
    fresh_gateway.con.execute("CREATE TEMP TABLE test_mut (id INT, name VARCHAR)")
    result = resource.execute_mutation("INSERT INTO test_mut VALUES (1, 'test') RETURNING id")

    # execute_mutation returns the result of fetchone()[0]
    assert result == EXPECTED_ONE


def test_storage_resource_execute_mutation_with_params(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource executes mutations with parameters."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    # Create a temp table
    fresh_gateway.con.execute("CREATE TEMP TABLE test_mut2 (id INT, name VARCHAR)")
    result = resource.execute_mutation(
        "INSERT INTO test_mut2 VALUES (?, ?) RETURNING id", [42, "test"]
    )

    # execute_mutation returns the result of fetchone()[0]
    assert result == EXPECTED_FORTY_TWO


def test_storage_resource_execute_mutation_multiple_rows(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource handles multi-row mutations."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    # Create a temp table and insert multiple rows
    fresh_gateway.con.execute("CREATE TEMP TABLE test_mut3 (id INT, name VARCHAR)")
    fresh_gateway.con.execute("INSERT INTO test_mut3 VALUES (1, 'a'), (2, 'b'), (3, 'c')")

    # Update multiple rows and return count
    result = resource.execute_mutation(
        "UPDATE test_mut3 SET name = 'updated' WHERE id > 0 RETURNING id"
    )

    # First row returned should be 1 (first updated id)
    assert result == EXPECTED_ONE


def test_storage_resource_protocol_conformance(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """StorageResource conforms to ResourceProvider protocol."""
    resource = StorageResource(gateway=fresh_gateway, _repo_root=tmp_path)

    assert isinstance(resource, ResourceProvider)


# ===========================================================================
# ResourceNotFoundError Tests
# ===========================================================================


def test_resource_not_found_error_message() -> None:
    """ResourceNotFoundError includes resource name."""
    error = ResourceNotFoundError("my_resource")

    assert "my_resource" in str(error)
    assert error.resource_name == "my_resource"


def test_resource_not_found_error_is_exception() -> None:
    """ResourceNotFoundError is an Exception."""
    error = ResourceNotFoundError("test")

    assert isinstance(error, Exception)
