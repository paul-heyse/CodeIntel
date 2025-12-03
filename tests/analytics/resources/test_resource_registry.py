"""Tests for the resource registry and provider protocol.

This module tests:
- ResourceRegistry for managing resource providers
- ResourceProvider protocol implementation
- ResourceNotFoundError and ResourceNotLoadedError
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.analytics.resources.protocol import (
    ResourceError,
    ResourceNotLoadedError,
)
from codeintel.analytics.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)


@dataclass
class MockProvider:
    """Mock resource provider for testing.

    Attributes
    ----------
    _resource
        The resource value to return.
    _loaded
        Whether the resource has been loaded.
    _name
        Name of the resource.
    """

    _resource: object
    _loaded: bool = False
    _name: str = "MockResource"

    @property
    def is_loaded(self) -> bool:
        """Check if resource is loaded.

        Returns
        -------
        bool
            True if loaded.
        """
        return self._loaded

    @property
    def resource_name(self) -> str:
        """Get resource name.

        Returns
        -------
        str
            Resource name.
        """
        return self._name

    def get(self) -> object:
        """Get the resource.

        Returns
        -------
        object
            The resource.

        Raises
        ------
        ResourceNotLoadedError
            If resource cannot be loaded.
        """
        if self._resource is None:
            raise ResourceNotLoadedError(self._name, "Resource is None")
        self._loaded = True
        return self._resource

    def get_or_none(self) -> object | None:
        """Get the resource or None.

        Returns
        -------
        object | None
            The resource or None.
        """
        try:
            return self.get()
        except ResourceNotLoadedError:
            return None

    def invalidate(self) -> None:
        """Invalidate the cached resource."""
        self._loaded = False


class MockResourceType:
    """Mock resource type for type-based registration."""


class AnotherResourceType:
    """Another mock resource type."""


def test_resource_registry_empty() -> None:
    """Empty registry has no providers."""
    registry = ResourceRegistry()
    assert len(registry.registered_types) == 0


def test_resource_registry_register() -> None:
    """Register a provider."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")

    registry.register(MockResourceType, provider)

    assert registry.has(MockResourceType)
    assert MockResourceType in registry.registered_types


def test_resource_registry_register_duplicate_raises() -> None:
    """Registering same type twice raises ValueError."""
    registry = ResourceRegistry()
    provider1 = MockProvider(_resource="first")
    provider2 = MockProvider(_resource="second")

    registry.register(MockResourceType, provider1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(MockResourceType, provider2)


def test_resource_registry_get() -> None:
    """Get a registered provider."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    result = registry.get(MockResourceType)

    assert result is provider


def test_resource_registry_get_not_found_raises() -> None:
    """Get unregistered type raises ResourceNotFoundError."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError) as exc_info:
        registry.get(MockResourceType)

    assert exc_info.value.resource_type is MockResourceType


def test_resource_registry_get_or_none() -> None:
    """Get or None returns provider when registered."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    result = registry.get_or_none(MockResourceType)

    assert result is provider


def test_resource_registry_get_or_none_not_found() -> None:
    """Get or None returns None when not registered."""
    registry = ResourceRegistry()

    result = registry.get_or_none(MockResourceType)

    assert result is None


def test_resource_registry_has() -> None:
    """Has returns True for registered types."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    assert registry.has(MockResourceType) is True
    assert registry.has(AnotherResourceType) is False


def test_resource_registry_register_or_replace() -> None:
    """Register or replace updates existing provider."""
    registry = ResourceRegistry()
    provider1 = MockProvider(_resource="first", _name="Provider1")
    provider2 = MockProvider(_resource="second", _name="Provider2")

    registry.register(MockResourceType, provider1)
    previous = registry.register_or_replace(MockResourceType, provider2)

    assert previous is provider1
    assert registry.get(MockResourceType) is provider2


def test_resource_registry_register_or_replace_new() -> None:
    """Register or replace returns None for new registration."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")

    previous = registry.register_or_replace(MockResourceType, provider)

    assert previous is None
    assert registry.has(MockResourceType)


def test_resource_registry_get_by_name() -> None:
    """Get provider by string name."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    result = registry.get_by_name("MockResourceType")

    assert result is provider


def test_resource_registry_get_by_name_custom() -> None:
    """Get provider by custom string name."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider, name="CustomName")

    result = registry.get_by_name("CustomName")

    assert result is provider


def test_resource_registry_get_by_name_not_found() -> None:
    """Get by name raises KeyError when not found."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError, match="not found by name"):
        registry.get_by_name("NonexistentType")


def test_resource_registry_has_by_name() -> None:
    """Has by name checks string name registration."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    assert registry.has_by_name("MockResourceType") is True
    assert registry.has_by_name("OtherName") is False


def test_resource_registry_require() -> None:
    """Require gets and loads the resource."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="resource_value")
    registry.register(MockResourceType, provider)

    result = registry.require(MockResourceType)

    assert result == "resource_value"
    assert provider.is_loaded


def test_resource_registry_require_not_found() -> None:
    """Require raises ResourceNotFoundError for unregistered type."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError):
        registry.require(MockResourceType)


def test_resource_registry_require_by_name() -> None:
    """Require by name gets and loads the resource."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="named_value")
    registry.register(MockResourceType, provider)

    result = registry.require_by_name("MockResourceType")

    assert result == "named_value"


def test_resource_registry_require_or_none() -> None:
    """Require or None returns resource when available."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="value")
    registry.register(MockResourceType, provider)

    result = registry.require_or_none(MockResourceType)

    assert result == "value"


def test_resource_registry_require_or_none_not_found() -> None:
    """Require or None returns None when not registered."""
    registry = ResourceRegistry()

    result = registry.require_or_none(MockResourceType)

    assert result is None


def test_resource_registry_require_or_none_not_loaded() -> None:
    """Require or None returns None when resource cannot load."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource=None)  # Will raise on get()
    registry.register(MockResourceType, provider)

    result = registry.require_or_none(MockResourceType)

    assert result is None


def test_resource_registry_invalidate_specific() -> None:
    """Invalidate specific resource type."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    # Load the resource
    registry.require(MockResourceType)
    assert provider.is_loaded

    # Invalidate
    registry.invalidate(MockResourceType)
    assert not provider.is_loaded


def test_resource_registry_invalidate_all() -> None:
    """Invalidate all resources."""
    registry = ResourceRegistry()
    provider1 = MockProvider(_resource="test1", _name="P1")
    provider2 = MockProvider(_resource="test2", _name="P2")
    registry.register(MockResourceType, provider1)
    registry.register(AnotherResourceType, provider2)

    # Load both
    registry.require(MockResourceType)
    registry.require(AnotherResourceType)
    assert provider1.is_loaded
    assert provider2.is_loaded

    # Invalidate all
    registry.invalidate()
    assert not provider1.is_loaded
    assert not provider2.is_loaded


def test_resource_registry_invalidate_not_registered() -> None:
    """Invalidate non-existent type does nothing."""
    registry = ResourceRegistry()

    # Should not raise
    registry.invalidate(MockResourceType)


def test_resource_registry_clear() -> None:
    """Clear removes all providers."""
    registry = ResourceRegistry()
    provider = MockProvider(_resource="test")
    registry.register(MockResourceType, provider)

    registry.clear()

    assert len(registry.registered_types) == 0
    assert not registry.has(MockResourceType)
    assert not registry.has_by_name("MockResourceType")


def test_resource_registry_registered_types() -> None:
    """Registered types returns all type keys."""
    registry = ResourceRegistry()
    registry.register(MockResourceType, MockProvider(_resource="1"))
    registry.register(AnotherResourceType, MockProvider(_resource="2"))

    types = registry.registered_types

    assert isinstance(types, frozenset)
    assert MockResourceType in types
    assert AnotherResourceType in types


def test_resource_not_found_error() -> None:
    """ResourceNotFoundError stores the resource type."""
    error = ResourceNotFoundError(MockResourceType)

    assert error.resource_type is MockResourceType
    assert "MockResourceType" in str(error)


def test_resource_not_loaded_error() -> None:
    """ResourceNotLoadedError stores type and reason."""
    error = ResourceNotLoadedError("TestResource", "connection failed")

    assert error.resource_type == "TestResource"
    assert error.reason == "connection failed"
    assert "TestResource" in str(error)
    assert "connection failed" in str(error)


def test_resource_not_loaded_error_no_reason() -> None:
    """ResourceNotLoadedError without reason."""
    error = ResourceNotLoadedError("TestResource")

    assert error.resource_type == "TestResource"
    assert error.reason is None
    assert "TestResource" in str(error)


def test_resource_error_hierarchy() -> None:
    """Resource errors inherit from ResourceError."""
    assert issubclass(ResourceNotLoadedError, ResourceError)


def test_mock_provider_protocol_compliance() -> None:
    """MockProvider implements ResourceProvider protocol."""
    provider = MockProvider(_resource="test", _name="TestProvider")

    assert hasattr(provider, "is_loaded")
    assert hasattr(provider, "resource_name")
    assert hasattr(provider, "get")
    assert hasattr(provider, "get_or_none")
    assert hasattr(provider, "invalidate")

    assert provider.resource_name == "TestProvider"
    assert provider.is_loaded is False

    value = provider.get()
    assert value == "test"
    assert provider.is_loaded is True

    provider.invalidate()
    assert provider.is_loaded is False


def test_resource_registry_multiple_registrations() -> None:
    """Register multiple different types."""
    registry = ResourceRegistry()

    for i in range(5):
        resource_type = type(f"ResourceType{i}", (), {})
        provider = MockProvider(_resource=f"value{i}", _name=f"Provider{i}")
        registry.register(resource_type, provider)

    expected_count = 5
    assert len(registry.registered_types) == expected_count
