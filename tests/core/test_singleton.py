"""Test SingletonHolder pattern from codeintel.core.singleton.

This module tests the thread-safe singleton holder pattern including:
- Singleton creation via factory functions
- Thread safety with double-checked locking
- Subclass isolation (independent singletons per subclass)
- Reset and initialization state tracking
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import pytest

from codeintel.core.singleton import SingletonHolder, SingletonNotInitializedError

# =============================================================================
# Test Fixture Classes
# =============================================================================


@dataclass
class SampleRegistry:
    """Sample registry class for singleton testing."""

    name: str
    items: list[str]

    @classmethod
    def create_default(cls) -> SampleRegistry:
        """Create a default registry instance."""
        return cls(name="default", items=[])


@dataclass
class AnotherRegistry:
    """Another registry class to test subclass isolation."""

    value: int


class SampleRegistryHolder(SingletonHolder[SampleRegistry]):
    """Holder for SampleRegistry singleton."""


class AnotherRegistryHolder(SingletonHolder[AnotherRegistry]):
    """Holder for AnotherRegistry singleton."""


class CountingHolder(SingletonHolder[int]):
    """Holder that tracks factory invocations."""

    call_count: ClassVar[int] = 0

    @classmethod
    def reset_count(cls) -> None:
        """Reset the call counter."""
        cls.call_count = 0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset all singleton holders before each test."""
    SampleRegistryHolder.reset()
    AnotherRegistryHolder.reset()
    CountingHolder.reset()
    CountingHolder.reset_count()


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_singleton_get_creates_instance_via_factory() -> None:
    """Verify that get() creates an instance using the factory function."""
    registry = SampleRegistryHolder.get(SampleRegistry.create_default)

    assert registry is not None
    assert isinstance(registry, SampleRegistry)
    assert registry.name == "default"
    assert registry.items == []


def test_singleton_get_returns_same_instance() -> None:
    """Verify that repeated get() calls return the same instance."""
    registry1 = SampleRegistryHolder.get(SampleRegistry.create_default)
    registry2 = SampleRegistryHolder.get(SampleRegistry.create_default)

    assert registry1 is registry2


def test_singleton_get_or_none_before_initialization() -> None:
    """Verify that get_or_none() returns None before initialization."""
    result = SampleRegistryHolder.get_or_none()

    assert result is None


def test_singleton_get_or_none_after_initialization() -> None:
    """Verify that get_or_none() returns the instance after initialization."""
    expected = SampleRegistryHolder.get(SampleRegistry.create_default)
    result = SampleRegistryHolder.get_or_none()

    assert result is expected


def test_singleton_is_initialized_false_initially() -> None:
    """Verify that is_initialized() returns False before get() is called."""
    assert not SampleRegistryHolder.is_initialized()


def test_singleton_is_initialized_true_after_get() -> None:
    """Verify that is_initialized() returns True after get() is called."""
    SampleRegistryHolder.get(SampleRegistry.create_default)

    assert SampleRegistryHolder.is_initialized()


def test_singleton_reset_clears_instance() -> None:
    """Verify that reset() clears the singleton instance."""
    SampleRegistryHolder.get(SampleRegistry.create_default)
    assert SampleRegistryHolder.is_initialized()

    SampleRegistryHolder.reset()

    assert not SampleRegistryHolder.is_initialized()
    assert SampleRegistryHolder.get_or_none() is None


def test_singleton_reset_allows_new_instance() -> None:
    """Verify that reset() allows a new instance to be created."""
    registry1 = SampleRegistryHolder.get(
        lambda: SampleRegistry(name="first", items=["a"])
    )
    SampleRegistryHolder.reset()
    registry2 = SampleRegistryHolder.get(
        lambda: SampleRegistry(name="second", items=["b"])
    )

    assert registry1 is not registry2
    assert registry1.name == "first"
    assert registry2.name == "second"


# =============================================================================
# Subclass Isolation Tests
# =============================================================================


def test_singleton_subclasses_are_independent() -> None:
    """Verify that different subclasses maintain independent singletons."""
    sample_registry = SampleRegistryHolder.get(SampleRegistry.create_default)
    another_registry = AnotherRegistryHolder.get(lambda: AnotherRegistry(value=42))

    assert isinstance(sample_registry, SampleRegistry)
    assert isinstance(another_registry, AnotherRegistry)
    assert SampleRegistryHolder.is_initialized()
    assert AnotherRegistryHolder.is_initialized()


def test_singleton_reset_only_affects_own_subclass() -> None:
    """Verify that reset() only affects the specific subclass."""
    SampleRegistryHolder.get(SampleRegistry.create_default)
    AnotherRegistryHolder.get(lambda: AnotherRegistry(value=42))

    SampleRegistryHolder.reset()

    assert not SampleRegistryHolder.is_initialized()
    assert AnotherRegistryHolder.is_initialized()


# =============================================================================
# Factory Invocation Tests
# =============================================================================


def test_singleton_factory_called_only_once() -> None:
    """Verify that the factory is only called once even with multiple get() calls."""
    call_count = 0

    def counting_factory() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    result1 = CountingHolder.get(counting_factory)
    result2 = CountingHolder.get(counting_factory)
    result3 = CountingHolder.get(counting_factory)

    assert result1 == result2 == result3 == 1
    assert call_count == 1


def test_singleton_factory_returning_none_raises_error() -> None:
    """Verify that a factory returning None raises SingletonNotInitializedError."""

    def null_factory() -> SampleRegistry | None:
        return None

    # The factory returns None, which should trigger the error after the factory runs
    # The SingletonHolder stores None, then checks and raises
    with pytest.raises(SingletonNotInitializedError) as exc_info:
        SampleRegistryHolder.get(null_factory)  # type: ignore[arg-type]

    assert "SampleRegistryHolder" in str(exc_info.value)


# =============================================================================
# Thread Safety Tests
# =============================================================================


def test_singleton_thread_safe_initialization() -> None:
    """Verify that singleton initialization is thread-safe."""
    results: list[SampleRegistry] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        try:
            barrier.wait()  # Synchronize all threads to start together
            registry = SampleRegistryHolder.get(SampleRegistry.create_default)
            results.append(registry)
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        for future in futures:
            future.result()

    assert not errors, f"Unexpected errors: {errors}"
    assert len(results) == 10
    # All results should be the same instance
    first = results[0]
    assert all(r is first for r in results)


def test_singleton_concurrent_get_single_factory_call() -> None:
    """Verify that concurrent get() calls only invoke factory once."""
    call_count = 0
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def counting_factory() -> int:
        nonlocal call_count
        with lock:
            call_count += 1
        return call_count

    def worker() -> int:
        barrier.wait()
        return CountingHolder.get(counting_factory)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        results = [f.result() for f in futures]

    # Factory should only be called once
    assert call_count == 1
    # All results should be 1 (the value from the single factory call)
    assert all(r == 1 for r in results)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_singleton_not_initialized_error_message() -> None:
    """Verify the error message format for SingletonNotInitializedError."""
    error = SingletonNotInitializedError("TestHolder")

    assert "TestHolder" in str(error)
    assert "not initialized" in str(error)


def test_singleton_factory_exception_propagates() -> None:
    """Verify that exceptions from the factory propagate correctly."""

    def failing_factory() -> SampleRegistry:
        msg = "Factory failed intentionally"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="Factory failed intentionally"):
        SampleRegistryHolder.get(failing_factory)

    # After a factory failure, the singleton should not be initialized
    assert not SampleRegistryHolder.is_initialized()
