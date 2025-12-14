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

if TYPE_CHECKING:
    from collections.abc import Mapping


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WorkerConfig()

        assert config.max_workers is None
        assert config.executor_type == "thread"
        assert config.env_var is None
        assert config.default_max == DEFAULT_MAX_WORKERS
        assert config.default_min == DEFAULT_MIN_WORKERS

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WorkerConfig(
            max_workers=8,
            executor_type="process",
            env_var="MY_WORKERS",
            default_max=32,
            default_min=4,
        )

        assert config.max_workers == 8
        assert config.executor_type == "process"
        assert config.env_var == "MY_WORKERS"
        assert config.default_max == 32
        assert config.default_min == 4

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = WorkerConfig()

        with pytest.raises(AttributeError):
            config.max_workers = 10  # type: ignore[misc]


class TestResolveWorkerCount:
    """Tests for resolve_worker_count function."""

    def test_explicit_count_takes_precedence(self) -> None:
        """Test that explicit count is used when provided."""
        result = resolve_worker_count(4)
        assert result == 4

    def test_explicit_count_over_env_var(self) -> None:
        """Test that explicit count takes precedence over env var."""
        env: Mapping[str, str] = {"MY_WORKERS": "16"}
        result = resolve_worker_count(4, env_var="MY_WORKERS", env=env)
        assert result == 4

    def test_env_var_override(self) -> None:
        """Test that env var is used when no explicit count."""
        env: Mapping[str, str] = {"MY_WORKERS": "12"}
        result = resolve_worker_count(env_var="MY_WORKERS", env=env)
        assert result == 12

    def test_invalid_env_var_ignored(self) -> None:
        """Test that invalid env var values are ignored."""
        env: Mapping[str, str] = {"MY_WORKERS": "not_a_number"}
        result = resolve_worker_count(env_var="MY_WORKERS", env=env)
        # Should fall back to CPU-based calculation
        assert result >= DEFAULT_MIN_WORKERS
        assert result <= DEFAULT_MAX_WORKERS

    def test_zero_explicit_count_uses_default(self) -> None:
        """Test that zero explicit count uses default."""
        result = resolve_worker_count(0)
        assert result >= DEFAULT_MIN_WORKERS

    def test_negative_explicit_count_uses_default(self) -> None:
        """Test that negative explicit count uses default."""
        result = resolve_worker_count(-1)
        assert result >= DEFAULT_MIN_WORKERS

    def test_custom_min_max(self) -> None:
        """Test custom min/max bounds."""
        result = resolve_worker_count(default_min=1, default_max=2)
        assert result >= 1
        assert result <= 2


class TestCreateExecutor:
    """Tests for create_executor function."""

    def test_thread_executor(self) -> None:
        """Test creating a thread pool executor."""
        config = WorkerConfig(max_workers=2, executor_type="thread")
        executor = create_executor(config)

        try:
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_process_executor(self) -> None:
        """Test creating a process pool executor."""
        config = WorkerConfig(max_workers=2, executor_type="process")
        executor = create_executor(config)

        try:
            assert isinstance(executor, ProcessPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_default_config(self) -> None:
        """Test creating executor with default config."""
        executor = create_executor()

        try:
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_env_var_in_config(self) -> None:
        """Test that env_var in config is respected."""
        config = WorkerConfig(env_var="TEST_WORKERS", executor_type="thread")
        executor = create_executor(config)

        try:
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)


class TestWorkerPool:
    """Tests for worker_pool context manager."""

    def test_thread_pool_context(self) -> None:
        """Test thread pool lifecycle in context manager."""
        with worker_pool("thread", 2) as executor:
            assert isinstance(executor, ThreadPoolExecutor)
            # Submit a simple task
            future = executor.submit(lambda: 42)
            assert future.result() == 42

    def test_process_pool_context(self) -> None:
        """Test process pool lifecycle in context manager."""
        with worker_pool("process", 2) as executor:
            assert isinstance(executor, ProcessPoolExecutor)

    def test_executor_shutdown_on_exit(self) -> None:
        """Test that executor is properly shut down on context exit."""
        with worker_pool("thread", 2) as executor:
            pass  # Exit the context

        # After context exit, submitting should fail
        with pytest.raises(RuntimeError):
            executor.submit(lambda: 42)


class TestExecutorFactory:
    """Tests for executor_factory function."""

    def test_creates_factory(self) -> None:
        """Test that factory function is returned."""
        factory = executor_factory("thread", 4)
        assert callable(factory)

    def test_factory_creates_thread_executor(self) -> None:
        """Test factory creates thread executor."""
        factory = executor_factory("thread", 2)
        executor = factory()

        try:
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_factory_creates_process_executor(self) -> None:
        """Test factory creates process executor."""
        factory = executor_factory("process", 2)
        executor = factory()

        try:
            assert isinstance(executor, ProcessPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_factory_creates_new_executor_each_call(self) -> None:
        """Test that each factory call creates a new executor."""
        factory = executor_factory("thread", 2)
        executor1 = factory()
        executor2 = factory()

        try:
            assert executor1 is not executor2
        finally:
            executor1.shutdown(wait=False)
            executor2.shutdown(wait=False)


class TestConstants:
    """Tests for module constants."""

    def test_default_max_workers(self) -> None:
        """Test DEFAULT_MAX_WORKERS is reasonable."""
        assert DEFAULT_MAX_WORKERS > 0
        assert DEFAULT_MAX_WORKERS == 16

    def test_default_min_workers(self) -> None:
        """Test DEFAULT_MIN_WORKERS is reasonable."""
        assert DEFAULT_MIN_WORKERS > 0
        assert DEFAULT_MIN_WORKERS == 2
        assert DEFAULT_MIN_WORKERS <= DEFAULT_MAX_WORKERS
