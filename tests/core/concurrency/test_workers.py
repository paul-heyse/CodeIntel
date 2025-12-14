"""Tests for core concurrency worker utilities.

This module tests worker pool configuration, creation, and lifecycle
management functions from codeintel.core.concurrency.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from codeintel.core.concurrency import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Test default configuration values."""
        config = WorkerConfig()

        expect_equal(config.max_workers, None)
        expect_equal(config.executor_type, "thread")
        expect_equal(config.env_var, None)
        expect_equal(config.default_max, DEFAULT_MAX_WORKERS)
        expect_equal(config.default_min, DEFAULT_MIN_WORKERS)

    @staticmethod
    def test_custom_values() -> None:
        """Test custom configuration values."""
        config = WorkerConfig(
            max_workers=8,
            executor_type="process",
            env_var="MY_WORKERS",
            default_max=32,
            default_min=4,
        )

        expect_equal(config.max_workers, 8)
        expect_equal(config.executor_type, "process")
        expect_equal(config.env_var, "MY_WORKERS")
        expect_equal(config.default_max, 32)
        expect_equal(config.default_min, 4)

    @staticmethod
    def test_frozen() -> None:
        """Test that config is immutable."""
        config = WorkerConfig()

        with pytest.raises(AttributeError):
            config.max_workers = 10  # type: ignore[misc]


class TestResolveWorkerCount:
    """Tests for resolve_worker_count function."""

    @staticmethod
    def test_explicit_count_takes_precedence() -> None:
        """Test that explicit count is used when provided."""
        result = resolve_worker_count(4)
        expect_equal(result, 4)

    @staticmethod
    def test_explicit_count_over_env_var() -> None:
        """Test that explicit count takes precedence over env var."""
        env: Mapping[str, str] = {"MY_WORKERS": "16"}
        result = resolve_worker_count(4, env_var="MY_WORKERS", env=env)
        expect_equal(result, 4)

    @staticmethod
    def test_env_var_override() -> None:
        """Test that env var is used when no explicit count."""
        env: Mapping[str, str] = {"MY_WORKERS": "12"}
        result = resolve_worker_count(env_var="MY_WORKERS", env=env)
        expect_equal(result, 12)

    @staticmethod
    def test_invalid_env_var_ignored() -> None:
        """Test that invalid env var values are ignored."""
        env: Mapping[str, str] = {"MY_WORKERS": "not_a_number"}
        result = resolve_worker_count(env_var="MY_WORKERS", env=env)
        # Should fall back to CPU-based calculation
        expect_true(result >= DEFAULT_MIN_WORKERS)
        expect_true(result <= DEFAULT_MAX_WORKERS)

    @staticmethod
    def test_zero_explicit_count_uses_default() -> None:
        """Test that zero explicit count uses default."""
        result = resolve_worker_count(0)
        expect_true(result >= DEFAULT_MIN_WORKERS)

    @staticmethod
    def test_negative_explicit_count_uses_default() -> None:
        """Test that negative explicit count uses default."""
        result = resolve_worker_count(-1)
        expect_true(result >= DEFAULT_MIN_WORKERS)

    @staticmethod
    def test_custom_min_max() -> None:
        """Test custom min/max bounds."""
        result = resolve_worker_count(default_min=1, default_max=2)
        expect_true(result >= 1)
        expect_true(result <= 2)


class TestCreateExecutor:
    """Tests for create_executor function."""

    @staticmethod
    def test_thread_executor() -> None:
        """Test creating a thread pool executor."""
        config = WorkerConfig(max_workers=2, executor_type="thread")
        executor = create_executor(config)

        try:
            expect_is_instance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def test_process_executor() -> None:
        """Test creating a process pool executor."""
        config = WorkerConfig(max_workers=2, executor_type="process")
        executor = create_executor(config)

        try:
            expect_is_instance(executor, ProcessPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def test_default_config() -> None:
        """Test creating executor with default config."""
        executor = create_executor()

        try:
            expect_is_instance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def test_env_var_in_config() -> None:
        """Test that env_var in config is respected."""
        config = WorkerConfig(env_var="TEST_WORKERS", executor_type="thread")
        executor = create_executor(config)

        try:
            expect_is_instance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)


class TestWorkerPool:
    """Tests for worker_pool context manager."""

    @staticmethod
    def test_thread_pool_context() -> None:
        """Test thread pool lifecycle in context manager."""
        with worker_pool("thread", 2) as executor:
            expect_is_instance(executor, ThreadPoolExecutor)
            # Submit a simple task
            future = executor.submit(lambda: 42)
            expect_equal(future.result(), 42)

    @staticmethod
    def test_process_pool_context() -> None:
        """Test process pool lifecycle in context manager."""
        with worker_pool("process", 2) as executor:
            expect_is_instance(executor, ProcessPoolExecutor)

    @staticmethod
    def test_executor_shutdown_on_exit() -> None:
        """Test that executor is properly shut down on context exit."""
        with worker_pool("thread", 2) as executor:
            pass  # Exit the context

        # After context exit, submitting should fail
        with pytest.raises(RuntimeError):
            executor.submit(lambda: 42)


class TestExecutorFactory:
    """Tests for executor_factory function."""

    @staticmethod
    def test_creates_factory() -> None:
        """Test that factory function is returned."""
        factory = executor_factory("thread", 4)
        expect_true(callable(factory))

    @staticmethod
    def test_factory_creates_thread_executor() -> None:
        """Test factory creates thread executor."""
        factory = executor_factory("thread", 2)
        executor = factory()

        try:
            expect_is_instance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def test_factory_creates_process_executor() -> None:
        """Test factory creates process executor."""
        factory = executor_factory("process", 2)
        executor = factory()

        try:
            expect_is_instance(executor, ProcessPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def test_factory_creates_new_executor_each_call() -> None:
        """Test that each factory call creates a new executor."""
        factory = executor_factory("thread", 2)
        executor1 = factory()
        executor2 = factory()

        try:
            expect_is_not(executor1, executor2)
        finally:
            executor1.shutdown(wait=False)
            executor2.shutdown(wait=False)


class TestConstants:
    """Tests for module constants."""

    @staticmethod
    def test_default_max_workers() -> None:
        """Test DEFAULT_MAX_WORKERS is reasonable."""
        expect_true(DEFAULT_MAX_WORKERS > 0)
        expect_equal(DEFAULT_MAX_WORKERS, 16)

    @staticmethod
    def test_default_min_workers() -> None:
        """Test DEFAULT_MIN_WORKERS is reasonable."""
        expect_true(DEFAULT_MIN_WORKERS > 0)
        expect_equal(DEFAULT_MIN_WORKERS, 2)
        expect_true(DEFAULT_MIN_WORKERS <= DEFAULT_MAX_WORKERS)
