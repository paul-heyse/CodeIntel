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
from typing import TYPE_CHECKING, ClassVar, cast

import pytest

from codeintel.core.singleton import (
    SingletonHolder,
    SingletonNotInitializedError,
    SingletonReentrancyError,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class SampleRegistry:
    """Sample registry class for singleton testing."""

    name: str
    items: list[str]

    @classmethod
    def create_default(cls) -> SampleRegistry:
        """Create a default registry instance.

        Returns
        -------
        SampleRegistry
            Registry with default values.
        """
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


class ReentrantHolder(SingletonHolder[int]):
    """Holder used to verify re-entrant initialization handling."""


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset all singleton holders before each test."""
    SampleRegistryHolder.reset()
    AnotherRegistryHolder.reset()
    CountingHolder.reset()
    CountingHolder.reset_count()
    ReentrantHolder.reset()


def test_singleton_get_creates_instance_via_factory() -> None:
    """Verify that get() creates an instance using the factory function."""
    registry = SampleRegistryHolder.get(SampleRegistry.create_default)

    expect_is_not_none(registry)
    expect_is_instance(registry, SampleRegistry)
    expect_equal(registry.name, "default")
    expect_equal(registry.items, [])


def test_singleton_get_returns_same_instance() -> None:
    """Verify that repeated get() calls return the same instance."""
    registry1 = SampleRegistryHolder.get(SampleRegistry.create_default)
    registry2 = SampleRegistryHolder.get(SampleRegistry.create_default)

    expect_true(registry1 is registry2)


def test_singleton_get_or_none_before_initialization() -> None:
    """Verify that get_or_none() returns None before initialization."""
    result = SampleRegistryHolder.get_or_none()

    expect_true(result is None)


def test_singleton_get_or_none_after_initialization() -> None:
    """Verify that get_or_none() returns the instance after initialization."""
    expected = SampleRegistryHolder.get(SampleRegistry.create_default)
    result = SampleRegistryHolder.get_or_none()

    expect_true(result is expected)


def test_singleton_is_initialized_false_initially() -> None:
    """Verify that is_initialized() returns False before get() is called."""
    expect_true(not SampleRegistryHolder.is_initialized())


def test_singleton_is_initialized_true_after_get() -> None:
    """Verify that is_initialized() returns True after get() is called."""
    SampleRegistryHolder.get(SampleRegistry.create_default)

    expect_true(SampleRegistryHolder.is_initialized())


def test_singleton_reset_clears_instance() -> None:
    """Verify that reset() clears the singleton instance."""
    SampleRegistryHolder.get(SampleRegistry.create_default)
    expect_true(SampleRegistryHolder.is_initialized())

    SampleRegistryHolder.reset()

    expect_true(not SampleRegistryHolder.is_initialized())
    expect_true(SampleRegistryHolder.get_or_none() is None)


def test_singleton_reset_allows_new_instance() -> None:
    """Verify that reset() allows a new instance to be created."""
    registry1 = SampleRegistryHolder.get(lambda: SampleRegistry(name="first", items=["a"]))
    SampleRegistryHolder.reset()
    registry2 = SampleRegistryHolder.get(lambda: SampleRegistry(name="second", items=["b"]))

    expect_true(registry1 is not registry2)
    expect_equal(registry1.name, "first")
    expect_equal(registry2.name, "second")


def test_singleton_subclasses_are_independent() -> None:
    """Verify that different subclasses maintain independent singletons."""
    sample_registry = SampleRegistryHolder.get(SampleRegistry.create_default)
    another_registry = AnotherRegistryHolder.get(lambda: AnotherRegistry(value=42))

    expect_is_instance(sample_registry, SampleRegistry)
    expect_is_instance(another_registry, AnotherRegistry)
    expect_true(SampleRegistryHolder.is_initialized())
    expect_true(AnotherRegistryHolder.is_initialized())


def test_singleton_reset_only_affects_own_subclass() -> None:
    """Verify that reset() only affects the specific subclass."""
    SampleRegistryHolder.get(SampleRegistry.create_default)
    AnotherRegistryHolder.get(lambda: AnotherRegistry(value=42))

    SampleRegistryHolder.reset()

    expect_true(not SampleRegistryHolder.is_initialized())
    expect_true(AnotherRegistryHolder.is_initialized())


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

    expect_true(result1 == result2 == result3 == 1)
    expect_equal(call_count, 1)


def test_singleton_factory_returning_none_raises_error() -> None:
    """Verify that a factory returning None raises SingletonNotInitializedError."""

    def null_factory() -> SampleRegistry | None:
        return None

    factory: Callable[[], SampleRegistry] = cast("Callable[[], SampleRegistry]", null_factory)

    with pytest.raises(SingletonNotInitializedError) as exc_info:
        SampleRegistryHolder.get(factory)

    expect_in("SampleRegistryHolder", str(exc_info.value))


def test_singleton_thread_safe_initialization() -> None:
    """Verify that singleton initialization is thread-safe."""
    results: list[SampleRegistry] = []
    errors: list[threading.BrokenBarrierError | RuntimeError] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        try:
            barrier.wait()
            registry = SampleRegistryHolder.get(SampleRegistry.create_default)
            results.append(registry)
        except (threading.BrokenBarrierError, RuntimeError) as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        for future in futures:
            future.result()

    expect_true(not errors, message=f"Unexpected errors: {errors}")
    expect_length(results, 10)

    first = results[0]
    expect_true(all(r is first for r in results))


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

    expect_equal(call_count, 1)

    expect_true(all(r == 1 for r in results))


def test_singleton_not_initialized_error_message() -> None:
    """Verify the error message format for SingletonNotInitializedError."""
    error = SingletonNotInitializedError("TestHolder")

    expect_in("TestHolder", str(error))
    expect_in("not initialized", str(error))


def test_singleton_factory_exception_propagates() -> None:
    """Verify that exceptions from the factory propagate correctly."""

    def failing_factory() -> SampleRegistry:
        msg = "Factory failed intentionally"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="Factory failed intentionally"):
        SampleRegistryHolder.get(failing_factory)

    expect_true(not SampleRegistryHolder.is_initialized())


def test_singleton_reentrancy_raises_error() -> None:
    """Verify re-entrant initialization raises SingletonReentrancyError."""

    def reentrant_factory() -> int:
        return ReentrantHolder.get(reentrant_factory)

    with pytest.raises(SingletonReentrancyError):
        ReentrantHolder.get(reentrant_factory)
