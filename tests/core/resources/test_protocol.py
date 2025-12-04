"""Test resource protocols from codeintel.core.resources.protocol.

This module tests:
- ResourceError, ResourceNotLoadedError exceptions
- ResourceProvider protocol check
- ResourceProviderBase caching behavior
- LazyResource get/get_or_none/invalidate/set_preloaded
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from codeintel.core.resources.protocol import (
    LazyResource,
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
)

# =============================================================================
# Test Implementations
# =============================================================================


class StringProvider(ResourceProviderBase[str]):
    """Provider that returns a string value."""

    RESOURCE_NAME: ClassVar[str] = "string_resource"

    def __init__(self, value: str) -> None:
        """Initialize with a value."""
        super().__init__()
        self._value = value

    def _load(self) -> str:
        """Load the string value."""
        return self._value


class CountingProvider(ResourceProviderBase[int]):
    """Provider that counts load calls."""

    RESOURCE_NAME: ClassVar[str] = "counting_resource"

    def __init__(self) -> None:
        """Initialize with zero load count."""
        super().__init__()
        self.load_count = 0

    def _load(self) -> int:
        """Load and count."""
        self.load_count += 1
        return self.load_count


class FailingProvider(ResourceProviderBase[str]):
    """Provider that fails on load."""

    RESOURCE_NAME: ClassVar[str] = "failing_resource"

    def _load(self) -> str:
        """Fail to load."""
        msg = "Load failed intentionally"
        raise ValueError(msg)


class LazyString(LazyResource[str]):
    """Lazy string resource for testing."""

    RESOURCE_NAME: ClassVar[str] = "lazy_string"

    def __init__(self, name: str, value: str) -> None:
        """Initialize with name and value."""
        super().__init__(name)
        self._value = value

    def _load(self) -> str:
        """Load the string value."""
        return self._value


class LazyFailer(LazyResource[str]):
    """Lazy resource that fails on load."""

    RESOURCE_NAME: ClassVar[str] = "lazy_failer"

    def __init__(self, name: str, error_msg: str) -> None:
        """Initialize with name and error message."""
        super().__init__(name)
        self._error_msg = error_msg

    def _load(self) -> str:
        """Fail to load."""
        raise ValueError(self._error_msg)


# =============================================================================
# ResourceError Tests
# =============================================================================


def test_resource_error_is_exception() -> None:
    """Verify ResourceError is an Exception."""
    error = ResourceError("Test error")

    assert isinstance(error, Exception)
    assert str(error) == "Test error"


def test_resource_error_can_be_raised() -> None:
    """Verify ResourceError can be raised and caught."""
    with pytest.raises(ResourceError) as exc_info:
        msg = "Test message"
        raise ResourceError(msg)

    assert "Test message" in str(exc_info.value)


# =============================================================================
# ResourceNotLoadedError Tests
# =============================================================================


def test_resource_not_loaded_error_message() -> None:
    """Verify ResourceNotLoadedError message format."""
    error = ResourceNotLoadedError("MyResource")

    assert "MyResource" in str(error)
    assert "not loaded" in str(error)
    assert error.resource_type == "MyResource"
    assert error.reason is None


def test_resource_not_loaded_error_with_reason() -> None:
    """Verify ResourceNotLoadedError includes reason."""
    error = ResourceNotLoadedError("MyResource", reason="File not found")

    assert "File not found" in str(error)
    assert error.reason == "File not found"


def test_resource_not_loaded_error_inheritance() -> None:
    """Verify ResourceNotLoadedError inherits from ResourceError."""
    error = ResourceNotLoadedError("Test")

    assert isinstance(error, ResourceError)
    assert isinstance(error, Exception)


# =============================================================================
# ResourceProvider Protocol Tests
# =============================================================================


def test_resource_provider_protocol_conformance() -> None:
    """Verify ResourceProviderBase implements ResourceProvider protocol."""
    provider = StringProvider("test")

    assert isinstance(provider, ResourceProvider)


def test_resource_provider_protocol_requires_get() -> None:
    """Verify ResourceProvider protocol requires get() method."""

    class MissingGet:
        RESOURCE_NAME: ClassVar[str] = "missing"

        def invalidate(self) -> None:
            pass

    # Missing get() method should not satisfy protocol
    assert not isinstance(MissingGet(), ResourceProvider)


def test_resource_provider_protocol_requires_invalidate() -> None:
    """Verify ResourceProvider protocol requires invalidate() method."""

    class MissingInvalidate:
        RESOURCE_NAME: ClassVar[str] = "missing"

        def get(self) -> str:
            return "test"

    # Missing invalidate() method should not satisfy protocol
    assert not isinstance(MissingInvalidate(), ResourceProvider)


# =============================================================================
# ResourceProviderBase Tests
# =============================================================================


def test_provider_base_get_returns_value() -> None:
    """Verify get() returns the loaded value."""
    provider = StringProvider("hello world")

    result = provider.get()

    assert result == "hello world"


def test_provider_base_get_caches_value() -> None:
    """Verify get() caches the loaded value."""
    provider = CountingProvider()

    # Multiple calls should return same value
    result1 = provider.get()
    result2 = provider.get()
    result3 = provider.get()

    assert result1 == result2 == result3 == 1
    assert provider.load_count == 1  # Only loaded once


def test_provider_base_invalidate_clears_cache() -> None:
    """Verify invalidate() clears the cached value."""
    provider = CountingProvider()

    provider.get()
    assert provider.load_count == 1

    provider.invalidate()
    provider.get()

    assert provider.load_count == 2


def test_provider_base_not_implemented_load() -> None:
    """Verify base class raises NotImplementedError for _load()."""
    provider = ResourceProviderBase[str]()

    with pytest.raises(NotImplementedError):
        provider.get()


def test_provider_base_resource_name() -> None:
    """Verify RESOURCE_NAME class attribute."""
    provider = StringProvider("test")

    assert provider.RESOURCE_NAME == "string_resource"


def test_provider_base_default_resource_name() -> None:
    """Verify default RESOURCE_NAME is empty string."""
    assert ResourceProviderBase.RESOURCE_NAME == ""


# =============================================================================
# LazyResource Tests
# =============================================================================


def test_lazy_resource_get() -> None:
    """Verify LazyResource.get() loads and returns value."""
    resource = LazyString("test", "lazy value")

    result = resource.get()

    assert result == "lazy value"


def test_lazy_resource_is_loaded_initially_false() -> None:
    """Verify is_loaded is False before first get()."""
    resource = LazyString("test", "value")

    assert resource.is_loaded is False


def test_lazy_resource_is_loaded_after_get() -> None:
    """Verify is_loaded is True after get()."""
    resource = LazyString("test", "value")

    resource.get()

    assert resource.is_loaded is True


def test_lazy_resource_caches_value() -> None:
    """Verify LazyResource caches loaded value."""
    load_count = [0]

    class CountingLazy(LazyResource[int]):
        def _load(self) -> int:
            load_count[0] += 1
            return load_count[0]

    resource = CountingLazy("counter")

    result1 = resource.get()
    result2 = resource.get()

    assert result1 == result2 == 1
    assert load_count[0] == 1


def test_lazy_resource_get_or_none_success() -> None:
    """Verify get_or_none() returns value on success."""
    resource = LazyString("test", "value")

    result = resource.get_or_none()

    assert result == "value"


def test_lazy_resource_get_or_none_on_failure() -> None:
    """Verify get_or_none() returns None on failure."""
    resource = LazyFailer("test", "intentional failure")

    result = resource.get_or_none()

    assert result is None


def test_lazy_resource_get_raises_on_failure() -> None:
    """Verify get() raises ResourceNotLoadedError on failure."""
    resource = LazyFailer("test", "load error")

    with pytest.raises(ResourceNotLoadedError) as exc_info:
        resource.get()

    assert "test" in str(exc_info.value)
    assert "load error" in str(exc_info.value)


def test_lazy_resource_get_raises_on_repeated_failure() -> None:
    """Verify get() raises on repeated calls after failure."""
    resource = LazyFailer("test", "persistent error")

    with pytest.raises(ResourceNotLoadedError):
        resource.get()

    # Second call should also raise (cached error)
    with pytest.raises(ResourceNotLoadedError):
        resource.get()


def test_lazy_resource_invalidate() -> None:
    """Verify invalidate() clears cached value and error."""
    resource = LazyString("test", "value")
    resource.get()
    assert resource.is_loaded is True

    resource.invalidate()

    assert resource.is_loaded is False


def test_lazy_resource_invalidate_clears_error() -> None:
    """Verify invalidate() clears cached error."""
    resource = LazyFailer("test", "error")

    with pytest.raises(ResourceNotLoadedError):
        resource.get()

    resource.invalidate()

    # After invalidate, should try to load again (and fail again)
    with pytest.raises(ResourceNotLoadedError):
        resource.get()


def test_lazy_resource_set_preloaded() -> None:
    """Verify set_preloaded() sets value without loading."""
    resource = LazyString("test", "should_not_load")

    resource.set_preloaded("preloaded_value")

    assert resource.is_loaded is True
    assert resource.get() == "preloaded_value"


def test_lazy_resource_set_preloaded_clears_error() -> None:
    """Verify set_preloaded() clears any cached error."""
    resource = LazyFailer("test", "error")

    with pytest.raises(ResourceNotLoadedError):
        resource.get()

    resource.set_preloaded("recovered")

    assert resource.get() == "recovered"


def test_lazy_resource_resource_name_property() -> None:
    """Verify resource_name property."""
    resource = LazyString("instance_name", "value")

    # Should prefer RESOURCE_NAME class var if set
    assert resource.resource_name == "lazy_string"


def test_lazy_resource_resource_name_fallback() -> None:
    """Verify resource_name falls back to instance name."""

    class NoResourceName(LazyResource[str]):
        RESOURCE_NAME = ""

        def _load(self) -> str:
            return "test"

    resource = NoResourceName("fallback_name")

    assert resource.resource_name == "fallback_name"


# =============================================================================
# Integration Tests
# =============================================================================


def test_provider_implements_protocol() -> None:
    """Verify custom providers implement ResourceProvider protocol."""
    providers = [
        StringProvider("test"),
        CountingProvider(),
    ]

    for provider in providers:
        assert isinstance(provider, ResourceProvider)
        assert hasattr(provider, "get")
        assert hasattr(provider, "invalidate")
        assert hasattr(provider, "RESOURCE_NAME")


def test_lazy_resource_as_provider() -> None:
    """Verify LazyResource can be used where ResourceProvider is expected."""
    resource = LazyString("test", "value")

    # Should have the required interface
    assert hasattr(resource, "get")
    assert hasattr(resource, "invalidate")
    assert hasattr(resource, "RESOURCE_NAME")
