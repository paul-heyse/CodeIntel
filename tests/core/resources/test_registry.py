"""Test resource registry from codeintel.core.resources.registry.

This module tests:
- ResourceNotFoundError with type and name
- ResourceRegistry register/register_or_replace/register_by_name
- register_provider() with RESOURCE_NAME
- register_factory() lazy instantiation
- get(), get_or_none(), get_by_name()
- require(), require_or_none(), require_by_name()
- has(), has_by_name() checks
- invalidate() single and all resources
- clear(), cleanup() methods
- registered_names, registered_types properties
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest

from codeintel.core.resources.protocol import LazyResource, ResourceProviderBase
from codeintel.core.resources.registry import (
    ResourceNotFoundError,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.core.resources.registry import (
        ResourceRegistry,
    )

INT_PROVIDER_VALUE = 42


class StringProvider(ResourceProviderBase[str]):
    """Provider that returns a string value."""

    RESOURCE_NAME: ClassVar[str] = "string_provider"

    def __init__(self, value: str) -> None:
        """Initialize with a value."""
        super().__init__()
        self._value = value

    def _load(self) -> str:
        """Load the string value.

        Returns
        -------
        str
            The stored string value.
        """
        return self._value


class IntProvider(ResourceProviderBase[int]):
    """Provider that returns an int value."""

    RESOURCE_NAME: ClassVar[str] = "int_provider"

    def __init__(self, value: int) -> None:
        """Initialize with a value."""
        super().__init__()
        self._value = value

    def _load(self) -> int:
        """Load the int value.

        Returns
        -------
        int
            The stored int value.
        """
        return self._value


class CountingProvider(ResourceProviderBase[int]):
    """Provider that increments each time it loads."""

    RESOURCE_NAME: ClassVar[str] = "counting_provider"

    def __init__(self) -> None:
        """Initialize with zero count."""
        super().__init__()
        self.load_count = 0

    def _load(self) -> int:
        """Increment and return the load count.

        Returns
        -------
        int
            The incremented count value.
        """
        self.load_count += 1
        return self.load_count


class LazyStringResource(LazyResource[str]):
    """Lazy string resource for testing."""

    RESOURCE_NAME: ClassVar[str] = "lazy_string"

    def __init__(self, name: str, value: str) -> None:
        """Initialize with name and value."""
        super().__init__(name)
        self._value = value

    def _load(self) -> str:
        """Load the string value.

        Returns
        -------
        str
            The stored string value.
        """
        return self._value


class SimpleValue:
    """Simple value object (not a provider)."""

    def __init__(self, value: str) -> None:
        """Initialize with value."""
        self.value = value


def test_resource_not_found_error_with_type() -> None:
    """Verify ResourceNotFoundError message with type."""
    error = ResourceNotFoundError(StringProvider)

    expect_true(error.resource_type is StringProvider)
    expect_equal(error.resource_name, "StringProvider")
    expect_in("StringProvider", str(error))


def test_resource_not_found_error_with_string() -> None:
    """Verify ResourceNotFoundError message with string name."""
    error = ResourceNotFoundError("custom_resource")

    expect_true(error.resource_type is None)
    expect_equal(error.resource_name, "custom_resource")
    expect_in("custom_resource", str(error))


def test_registry_register(resource_registry: ResourceRegistry) -> None:
    """Verify register() adds a provider."""
    provider = StringProvider("test")

    resource_registry.register(StringProvider, provider)

    expect_true(resource_registry.has(StringProvider))


def test_registry_register_duplicate_raises(resource_registry: ResourceRegistry) -> None:
    """Verify registering duplicate raises ValueError."""
    provider = StringProvider("test")
    resource_registry.register(StringProvider, provider)

    with pytest.raises(ValueError, match="already registered"):
        resource_registry.register(StringProvider, StringProvider("other"))


def test_registry_register_by_name(resource_registry: ResourceRegistry) -> None:
    """Verify register_by_name() adds provider by string name."""
    provider = StringProvider("test")

    resource_registry.register_by_name("my_string", provider)

    expect_true(resource_registry.has_by_name("my_string"))


def test_registry_register_or_replace(resource_registry: ResourceRegistry) -> None:
    """Verify register_or_replace() replaces existing provider."""
    provider1 = StringProvider("first")
    provider2 = StringProvider("second")

    resource_registry.register(StringProvider, provider1)
    previous = resource_registry.register_or_replace(StringProvider, provider2)

    expect_true(previous is provider1)
    expect_true(resource_registry.get(StringProvider) is provider2)


def test_registry_register_or_replace_returns_none(
    resource_registry: ResourceRegistry,
) -> None:
    """Verify register_or_replace() returns None when no previous."""
    provider = StringProvider("test")

    previous = resource_registry.register_or_replace(StringProvider, provider)

    expect_true(previous is None)


def test_registry_register_provider(resource_registry: ResourceRegistry) -> None:
    """Verify register_provider() uses RESOURCE_NAME."""
    provider = StringProvider("test")

    resource_registry.register_provider(provider)

    expect_true(resource_registry.has_by_name("string_provider"))


def test_registry_register_provider_without_name_raises(
    resource_registry: ResourceRegistry,
) -> None:
    """Verify register_provider() raises for missing RESOURCE_NAME."""

    class NoNameProvider:
        pass

    with pytest.raises(ValueError, match="RESOURCE_NAME"):
        resource_registry.register_provider(NoNameProvider())


def test_registry_register_with_custom_name(resource_registry: ResourceRegistry) -> None:
    """Verify register() accepts custom name."""
    provider = StringProvider("test")

    resource_registry.register(StringProvider, provider, name="custom_name")

    expect_true(resource_registry.has_by_name("custom_name"))


def test_registry_register_factory(resource_registry: ResourceRegistry) -> None:
    """Verify register_factory() registers lazy factory."""
    call_count = [0]

    def factory() -> StringProvider:
        call_count[0] += 1
        return StringProvider("from_factory")

    resource_registry.register_factory("lazy_provider", factory)

    expect_equal(call_count[0], 0)

    provider = resource_registry.get_by_name("lazy_provider")

    expect_equal(call_count[0], 1)
    expect_true(isinstance(provider, StringProvider))


def test_registry_factory_called_once(resource_registry: ResourceRegistry) -> None:
    """Verify factory is only called once."""
    call_count = [0]

    def factory() -> StringProvider:
        call_count[0] += 1
        return StringProvider("cached")

    resource_registry.register_factory("lazy", factory)

    resource_registry.get_by_name("lazy")
    resource_registry.get_by_name("lazy")
    resource_registry.get_by_name("lazy")

    expect_equal(call_count[0], 1)


def test_registry_has_by_name_includes_factories(
    resource_registry: ResourceRegistry,
) -> None:
    """Verify has_by_name() returns True for registered factories."""
    resource_registry.register_factory("pending", lambda: StringProvider("x"))

    expect_true(resource_registry.has_by_name("pending"))


def test_registry_get(resource_registry: ResourceRegistry) -> None:
    """Verify get() returns the registered provider."""
    provider = StringProvider("test")
    resource_registry.register(StringProvider, provider)

    result = resource_registry.get(StringProvider)

    expect_true(result is provider)


def test_registry_get_missing_raises(resource_registry: ResourceRegistry) -> None:
    """Verify get() raises ResourceNotFoundError for missing."""
    with pytest.raises(ResourceNotFoundError):
        resource_registry.get(StringProvider)


def test_registry_get_or_none(resource_registry: ResourceRegistry) -> None:
    """Verify get_or_none() returns provider when present."""
    provider = StringProvider("test")
    resource_registry.register(StringProvider, provider)

    result = resource_registry.get_or_none(StringProvider)

    expect_true(result is provider)


def test_registry_get_or_none_missing(resource_registry: ResourceRegistry) -> None:
    """Verify get_or_none() returns None when missing."""
    result = resource_registry.get_or_none(StringProvider)

    expect_true(result is None)


def test_registry_get_by_name(resource_registry: ResourceRegistry) -> None:
    """Verify get_by_name() returns provider."""
    provider = StringProvider("test")
    resource_registry.register_by_name("my_provider", provider)

    result = resource_registry.get_by_name("my_provider")

    expect_true(result is provider)


def test_registry_get_by_name_missing_raises(resource_registry: ResourceRegistry) -> None:
    """Verify get_by_name() raises KeyError for missing."""
    with pytest.raises(KeyError, match="not found"):
        resource_registry.get_by_name("nonexistent")


def test_registry_require(resource_registry: ResourceRegistry) -> None:
    """Verify require() returns provider value."""
    provider = StringProvider("test_value")
    resource_registry.register(StringProvider, provider)

    result = resource_registry.require(StringProvider)

    expect_equal(result, "test_value")


def test_registry_require_missing_raises(resource_registry: ResourceRegistry) -> None:
    """Verify require() raises for missing resource."""
    with pytest.raises(ResourceNotFoundError):
        resource_registry.require(StringProvider)


def test_registry_require_or_none(resource_registry: ResourceRegistry) -> None:
    """Verify require_or_none() returns value when present."""
    provider = IntProvider(INT_PROVIDER_VALUE)
    resource_registry.register(IntProvider, provider)

    result = resource_registry.require_or_none(IntProvider)

    expect_equal(result, INT_PROVIDER_VALUE)


def test_registry_require_or_none_missing(resource_registry: ResourceRegistry) -> None:
    """Verify require_or_none() returns None when missing."""
    result = resource_registry.require_or_none(StringProvider)

    expect_true(result is None)


def test_registry_require_non_gettable(resource_registry: ResourceRegistry) -> None:
    """Verify require() returns provider directly if no get() method."""
    value = SimpleValue("direct")
    resource_registry.register(SimpleValue, value)

    result = resource_registry.require(SimpleValue)

    expect_true(isinstance(result, SimpleValue))


def test_registry_require_by_name(resource_registry: ResourceRegistry) -> None:
    """Verify require_by_name() returns provider value."""
    provider = StringProvider("named_value")
    resource_registry.register_by_name("named", provider)

    result = resource_registry.require_by_name("named")

    expect_equal(result, "named_value")


def test_registry_has_true(resource_registry: ResourceRegistry) -> None:
    """Verify has() returns True when registered."""
    resource_registry.register(StringProvider, StringProvider("test"))

    expect_true(resource_registry.has(StringProvider))


def test_registry_has_false(resource_registry: ResourceRegistry) -> None:
    """Verify has() returns False when not registered."""
    expect_true(not resource_registry.has(StringProvider))


def test_registry_has_by_name_true(resource_registry: ResourceRegistry) -> None:
    """Verify has_by_name() returns True when registered."""
    resource_registry.register_by_name("test", StringProvider("test"))

    expect_true(resource_registry.has_by_name("test"))


def test_registry_has_by_name_false(resource_registry: ResourceRegistry) -> None:
    """Verify has_by_name() returns False when not registered."""
    expect_true(not resource_registry.has_by_name("nonexistent"))


def test_registry_contains(resource_registry: ResourceRegistry) -> None:
    """Verify __contains__ works with 'in' operator."""
    resource_registry.register(StringProvider, StringProvider("test"))

    expect_true(StringProvider in resource_registry)
    expect_true(IntProvider not in resource_registry)


def test_registry_invalidate_single(resource_registry: ResourceRegistry) -> None:
    """Verify invalidate() invalidates single resource."""
    provider = CountingProvider()
    resource_registry.register(CountingProvider, provider)

    resource_registry.require(CountingProvider)
    first_count = provider.load_count

    resource_registry.invalidate(CountingProvider)

    resource_registry.require(CountingProvider)
    second_count = provider.load_count

    expect_equal(second_count, first_count + 1)


def test_registry_invalidate_all(resource_registry: ResourceRegistry) -> None:
    """Verify invalidate() without args invalidates all resources."""
    provider1 = CountingProvider()
    provider2 = IntProvider(INT_PROVIDER_VALUE)
    resource_registry.register(CountingProvider, provider1)
    resource_registry.register(IntProvider, provider2)

    resource_registry.require(CountingProvider)
    first_count = provider1.load_count
    resource_registry.require(IntProvider)

    resource_registry.invalidate()

    resource_registry.require(CountingProvider)
    refreshed_count = provider1.load_count
    expect_equal(refreshed_count, first_count + 1)
    expect_equal(provider2.get(), INT_PROVIDER_VALUE)


def test_registry_invalidate_missing_no_error(
    resource_registry: ResourceRegistry,
) -> None:
    """Verify invalidate() doesn't raise for missing resource."""
    resource_registry.invalidate(StringProvider)


def test_registry_clear(resource_registry: ResourceRegistry) -> None:
    """Verify clear() removes all providers."""
    resource_registry.register(StringProvider, StringProvider("test"))
    resource_registry.register(IntProvider, IntProvider(42))
    resource_registry.register_factory("lazy", lambda: StringProvider("x"))

    resource_registry.clear()

    expect_equal(len(resource_registry), 0)
    expect_true(not resource_registry.has(StringProvider))
    expect_true(not resource_registry.has_by_name("lazy"))


def test_registry_cleanup(resource_registry: ResourceRegistry) -> None:
    """Verify cleanup() invalidates and clears."""
    provider = CountingProvider()
    resource_registry.register(CountingProvider, provider)
    resource_registry.require(CountingProvider)
    first_count = provider.load_count

    resource_registry.cleanup()

    reloaded_value = provider.get()
    expect_equal(reloaded_value, first_count + 1)
    expect_equal(len(resource_registry), 0)


def test_registry_registered_names(resource_registry: ResourceRegistry) -> None:
    """Verify registered_names returns all names."""
    resource_registry.register_by_name("alpha", StringProvider("a"))
    resource_registry.register_by_name("beta", IntProvider(1))
    resource_registry.register_factory("gamma", lambda: StringProvider("g"))

    names = resource_registry.registered_names

    expect_in("alpha", names)
    expect_in("beta", names)
    expect_in("gamma", names)


def test_registry_registered_types(resource_registry: ResourceRegistry) -> None:
    """Verify registered_types returns all types."""
    resource_registry.register(StringProvider, StringProvider("test"))
    resource_registry.register(IntProvider, IntProvider(42))

    types = resource_registry.registered_types

    expect_true(StringProvider in types)
    expect_true(IntProvider in types)
    expect_true(isinstance(types, frozenset))


def test_registry_len(resource_registry: ResourceRegistry) -> None:
    """Verify __len__ returns provider count."""
    initial_count = len(resource_registry)

    resource_registry.register(StringProvider, StringProvider("test"))
    expect_equal(len(resource_registry), initial_count + 1)

    resource_registry.register(IntProvider, IntProvider(42))
    expect_equal(len(resource_registry), initial_count + 2)


def test_registry_with_lazy_resource(resource_registry: ResourceRegistry) -> None:
    """Verify registry works with LazyResource."""
    lazy = LazyStringResource("test", "lazy_value")
    resource_registry.register(LazyStringResource, lazy)

    provider = resource_registry.get(LazyStringResource)
    expect_true(provider is lazy)

    value = resource_registry.require(LazyStringResource)
    expect_equal(value, "lazy_value")


def test_registry_full_lifecycle(resource_registry: ResourceRegistry) -> None:
    """Verify full registry lifecycle."""
    provider = CountingProvider()
    resource_registry.register(CountingProvider, provider, name="lifecycle_test")

    expect_true(resource_registry.has(CountingProvider))
    expect_true(resource_registry.has_by_name("lifecycle_test"))
    expect_true(CountingProvider in resource_registry)

    expect_true(resource_registry.get(CountingProvider) is provider)
    resource_registry.require(CountingProvider)
    first_count = provider.load_count

    resource_registry.invalidate(CountingProvider)
    resource_registry.require(CountingProvider)
    expect_equal(provider.load_count, first_count + 1)

    new_provider = StringProvider("replaced")
    resource_registry.register_or_replace(StringProvider, new_provider)
    expect_equal(resource_registry.require(StringProvider), "replaced")

    resource_registry.cleanup()
    expect_equal(len(resource_registry), 0)
