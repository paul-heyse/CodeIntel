"""Tests for ingestion resource registry and providers.

This module tests the resource registry and provider infrastructure
used to manage dependencies during ingestion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from codeintel.ingestion.resources.protocol import ResourceError, ResourceProvider
from codeintel.ingestion.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)

# Test constants
TEST_COUNT_42 = 42
TEST_COUNT_100 = 100
EXPECTED_LEN_3 = 3


# =============================================================================
# Test Providers
# =============================================================================


@dataclass
class TestResource:
    """A simple test resource."""

    value: str
    count: int


class TestProvider(ResourceProvider[TestResource]):
    """Provider that returns a TestResource."""

    def __init__(self, value: str = "test", count: int = 10) -> None:
        """Initialize with test data."""
        self._value = value
        self._count = count
        self._loaded = False

    def get(self) -> TestResource:
        """Return the test resource.

        Returns
        -------
        TestResource
            The test resource instance.
        """
        self._loaded = True
        return TestResource(value=self._value, count=self._count)

    @property
    def is_loaded(self) -> bool:
        """Return whether the resource has been loaded."""
        return self._loaded

    def get_or_none(self) -> TestResource | None:
        """Return the resource without marking it loaded."""
        return TestResource(value=self._value, count=self._count)

    @property
    def resource_name(self) -> str:
        """Return the registry key for this resource."""
        return "test_resource"

    def invalidate(self) -> None:
        """Reset the loaded flag."""
        self._loaded = False


class ListProvider(ResourceProvider[Sequence[str]]):
    """Provider that returns a list of strings."""

    def __init__(self, items: Sequence[str] | None = None) -> None:
        """Initialize with optional items."""
        self._items = list(items or [])
        self._loaded = False

    def get(self) -> Sequence[str]:
        """Return the list of items.

        Returns
        -------
        Sequence[str]
            The list of items.
        """
        self._loaded = True
        return self._items

    @property
    def is_loaded(self) -> bool:
        """Return whether the items have been fetched."""
        return self._loaded

    def get_or_none(self) -> Sequence[str] | None:
        """Return items without forcing load semantics."""
        return self._items

    @property
    def resource_name(self) -> str:
        """Return the registry key for this resource."""
        return "list_resource"

    def invalidate(self) -> None:
        """Reset the loaded flag."""
        self._loaded = False


class FailingProvider(ResourceProvider[str]):
    """Provider that always fails."""

    def get(self) -> str:
        """Raise an error on access.

        Raises
        ------
        ResourceError
            Always raised.
        """
        _ = self  # Use self for PLR6301
        msg = "Resource unavailable"
        raise ResourceError(msg)

    @property
    def is_loaded(self) -> bool:
        """Return whether the resource has been loaded."""
        return False

    def get_or_none(self) -> str | None:
        """Return None because the resource always fails."""
        return None

    @property
    def resource_name(self) -> str:
        """Return the registry key for this resource."""
        return "failing_resource"

    def invalidate(self) -> None:
        """Failing provider has no cache to clear."""
        return


# =============================================================================
# ResourceNotFoundError Tests
# =============================================================================


def test_resource_not_found_error_from_type() -> None:
    """ResourceNotFoundError should accept a type."""
    error = ResourceNotFoundError(TestProvider)

    assert "TestProvider" in str(error)
    assert error.resource_type_name == "TestProvider"


def test_resource_not_found_error_from_string() -> None:
    """ResourceNotFoundError should accept a string."""
    error = ResourceNotFoundError("custom_resource")

    assert "custom_resource" in str(error)
    assert error.resource_type_name == "custom_resource"


def test_resource_not_found_inherits_resource_error() -> None:
    """ResourceNotFoundError should inherit from ResourceError."""
    error = ResourceNotFoundError(TestProvider)

    assert isinstance(error, ResourceError)


# =============================================================================
# ResourceRegistry Basic Tests
# =============================================================================


def test_registry_init() -> None:
    """ResourceRegistry should initialize."""
    registry = ResourceRegistry()

    assert registry is not None


def test_registry_register_and_get() -> None:
    """ResourceRegistry should register and retrieve providers."""
    registry = ResourceRegistry()
    provider = TestProvider()

    registry.register(TestProvider, provider)

    retrieved = registry.get(TestProvider)
    assert retrieved is provider


def test_registry_get_resource_value() -> None:
    """Registry providers should return their resource value."""
    registry = ResourceRegistry()
    provider = TestProvider(value="hello", count=TEST_COUNT_42)
    registry.register(TestProvider, provider)

    retrieved = registry.get(TestProvider)
    resource = retrieved.get()

    assert resource.value == "hello"
    assert resource.count == TEST_COUNT_42


def test_registry_get_nonexistent_raises() -> None:
    """ResourceRegistry.get should raise for unknown types."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError, match="TestProvider"):
        registry.get(TestProvider)


def test_registry_duplicate_registration_raises() -> None:
    """ResourceRegistry should reject duplicate registrations."""
    registry = ResourceRegistry()
    provider1 = TestProvider()
    provider2 = TestProvider()

    registry.register(TestProvider, provider1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(TestProvider, provider2)


def test_registry_register_or_replace() -> None:
    """ResourceRegistry.register_or_replace should allow replacement."""
    registry = ResourceRegistry()
    provider1 = TestProvider(value="first")
    provider2 = TestProvider(value="second")

    registry.register(TestProvider, provider1)
    registry.register_or_replace(TestProvider, provider2)

    retrieved = registry.get(TestProvider)
    assert retrieved.get().value == "second"


# =============================================================================
# ResourceRegistry Name-based Access Tests
# =============================================================================


def test_registry_get_by_name() -> None:
    """ResourceRegistry should support name-based lookup."""
    registry = ResourceRegistry()
    provider = TestProvider()

    registry.register(TestProvider, provider)

    retrieved = registry.get_by_name("TestProvider")
    assert retrieved is provider


def test_registry_get_by_name_nonexistent_raises() -> None:
    """ResourceRegistry.get_by_name should raise for unknown names."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError, match="unknown_name"):
        registry.get_by_name("unknown_name")


# =============================================================================
# ResourceRegistry Has Tests
# =============================================================================


def test_registry_has_registered() -> None:
    """ResourceRegistry.has should return True for registered types."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    assert registry.has(TestProvider) is True


def test_registry_has_unregistered() -> None:
    """ResourceRegistry.has should return False for unregistered types."""
    registry = ResourceRegistry()

    assert registry.has(TestProvider) is False


def test_registry_has_by_name() -> None:
    """ResourceRegistry.has_by_name should check name existence."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    assert registry.has_by_name("TestProvider") is True
    assert registry.has_by_name("unknown") is False


# =============================================================================
# ResourceRegistry Get Or None Tests
# =============================================================================


def test_registry_get_or_none_found() -> None:
    """ResourceRegistry.get_or_none should return provider when found."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    result = registry.get_or_none(TestProvider)

    assert result is provider


def test_registry_get_or_none_not_found() -> None:
    """ResourceRegistry.get_or_none should return None when not found."""
    registry = ResourceRegistry()

    result = registry.get_or_none(TestProvider)

    assert result is None


# =============================================================================
# ResourceRegistry Require Tests
# =============================================================================


def test_registry_require_found() -> None:
    """ResourceRegistry.require should return resource value when found."""
    registry = ResourceRegistry()
    provider = TestProvider(value="test_value")
    registry.register(TestProvider, provider)

    result = registry.require(TestProvider)

    # require returns the resource value, not the provider
    assert isinstance(result, TestResource)
    assert result.value == "test_value"


def test_registry_require_not_found() -> None:
    """ResourceRegistry.require should raise when not found."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError):
        registry.require(TestProvider)


def test_registry_require_by_name_found() -> None:
    """ResourceRegistry.require_by_name should return resource when found."""
    registry = ResourceRegistry()
    provider = TestProvider(value="named_value")
    registry.register(TestProvider, provider)

    result = registry.require_by_name("TestProvider")

    # require_by_name returns the resource value
    assert isinstance(result, TestResource)
    assert result.value == "named_value"


# =============================================================================
# ResourceRegistry Clear Tests
# =============================================================================


def test_registry_clear() -> None:
    """ResourceRegistry.clear should remove all providers."""
    registry = ResourceRegistry()
    registry.register(TestProvider, TestProvider())
    registry.register(ListProvider, ListProvider())

    registry.clear()

    assert registry.has(TestProvider) is False
    assert registry.has(ListProvider) is False


# =============================================================================
# ResourceRegistry Multiple Providers Tests
# =============================================================================


def test_registry_multiple_different_types() -> None:
    """ResourceRegistry should handle multiple provider types."""
    registry = ResourceRegistry()
    test_provider = TestProvider(value="test")
    list_provider = ListProvider(items=["a", "b", "c"])

    registry.register(TestProvider, test_provider)
    registry.register(ListProvider, list_provider)

    assert registry.get(TestProvider).get().value == "test"
    items = registry.get(ListProvider).get()
    assert len(items) == EXPECTED_LEN_3


# =============================================================================
# ResourceProvider Tests
# =============================================================================


def test_provider_resource_name_attribute() -> None:
    """ResourceProvider should have resource_name attribute."""
    provider = TestProvider()

    assert provider.resource_name == "test_resource"


def test_failing_provider_raises() -> None:
    """FailingProvider should raise ResourceError."""
    provider = FailingProvider()

    with pytest.raises(ResourceError, match="unavailable"):
        provider.get()


# =============================================================================
# Integration Tests
# =============================================================================


def test_registry_workflow() -> None:
    """Test typical registry workflow."""
    # Create registry
    registry = ResourceRegistry()

    # Register providers
    test_provider = TestProvider(value="hello", count=TEST_COUNT_100)
    list_provider = ListProvider(items=["x", "y"])

    registry.register(TestProvider, test_provider)
    registry.register(ListProvider, list_provider)

    # Use providers
    test_resource = registry.get(TestProvider).get()
    list_resource = registry.get(ListProvider).get()

    assert test_resource.value == "hello"
    assert test_resource.count == TEST_COUNT_100
    assert list(list_resource) == ["x", "y"]


def test_registry_optional_access() -> None:
    """Registry should support optional access pattern."""
    registry = ResourceRegistry()

    # Check before access using has()
    if registry.has(TestProvider):
        provider = registry.get(TestProvider)
        _ = provider.get()
    else:
        # Provider not registered - expected path
        pass

    # Now register and access
    registry.register(TestProvider, TestProvider())
    assert registry.has(TestProvider)


def test_registry_registered_types() -> None:
    """Registry should report registered types."""
    registry = ResourceRegistry()
    registry.register(TestProvider, TestProvider())
    registry.register(ListProvider, ListProvider())

    # registered_types is a property, not a method
    types = registry.registered_types

    assert TestProvider in types
    assert ListProvider in types
