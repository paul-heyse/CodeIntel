"""Parallel execution adapters for Hamilton builds.

This module provides adapters for parallel execution of Hamilton DAGs,
enabling better resource utilization for I/O-bound workloads.

Supported backends:
- threadpool: Multi-threaded execution using ThreadPoolExecutor
- sequential: Default single-threaded execution (no adapter needed)
- auto: Automatically select best available backend

Examples
--------
Create a threadpool adapter:

>>> from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
>>> adapter = create_parallel_adapter("threadpool", max_workers=4)
>>> dr = driver.Builder().with_adapters(adapter).build()

Check available backends:

>>> from codeintel.build.hamilton.adapters.parallel import get_available_backends
>>> backends = get_available_backends()
>>> print(backends)  # ['sequential', 'threadpool']
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hamilton.lifecycle import ResultBuilder

__all__ = [
    "ExecutionBackend",
    "ParallelConfig",
    "ThreadPoolAdapter",
    "create_parallel_adapter",
    "get_available_backends",
]

log = logging.getLogger(__name__)


class ExecutionBackend(Enum):
    """Available execution backends.

    Attributes
    ----------
    SEQUENTIAL
        Default single-threaded execution.
    THREADPOOL
        Multi-threaded execution using ThreadPoolExecutor.
    AUTO
        Automatically select best available backend.
    """

    SEQUENTIAL = "sequential"
    THREADPOOL = "threadpool"
    AUTO = "auto"


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes
    ----------
    backend
        The execution backend to use.
    max_workers
        Maximum number of parallel workers.
        For threadpool: number of threads.
    thread_name_prefix
        Prefix for thread names (threadpool only).

    Examples
    --------
    >>> config = ParallelConfig(
    ...     backend=ExecutionBackend.THREADPOOL,
    ...     max_workers=8,
    ... )
    """

    backend: ExecutionBackend = ExecutionBackend.SEQUENTIAL
    max_workers: int | None = None
    thread_name_prefix: str = "hamilton-build"

    @classmethod
    def from_env(cls) -> ParallelConfig:
        """Create config from environment variables.

        Reads:
        - HAMILTON_BACKEND: Backend name (sequential, threadpool, auto)
        - HAMILTON_MAX_WORKERS: Number of workers

        Returns
        -------
        ParallelConfig
            Configuration from environment.
        """
        backend_str = os.getenv("HAMILTON_BACKEND", "sequential").lower()
        try:
            backend = ExecutionBackend(backend_str)
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend_str)
            backend = ExecutionBackend.SEQUENTIAL

        max_workers_str = os.getenv("HAMILTON_MAX_WORKERS")
        max_workers = int(max_workers_str) if max_workers_str else None

        return cls(backend=backend, max_workers=max_workers)

    @classmethod
    def from_cli_args(
        cls,
        backend: str | None = None,
        max_workers: int | None = None,
    ) -> ParallelConfig:
        """Create config from CLI arguments.

        Parameters
        ----------
        backend
            Backend name from CLI.
        max_workers
            Number of workers from CLI.

        Returns
        -------
        ParallelConfig
            Configuration from CLI arguments.
        """
        if backend is None:
            return cls.from_env()

        try:
            backend_enum = ExecutionBackend(backend.lower())
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend)
            backend_enum = ExecutionBackend.SEQUENTIAL

        return cls(backend=backend_enum, max_workers=max_workers)


def get_available_backends() -> list[str]:
    """Get list of available execution backends.

    Returns
    -------
    list[str]
        Names of available backends.
    """
    available = ["sequential", "threadpool"]  # Always available

    # Check for optional backends (future support)
    try:
        import ray  # noqa: F401

        available.append("ray")
    except ImportError:
        pass

    try:
        import dask  # noqa: F401

        available.append("dask")
    except ImportError:
        pass

    return available


class ThreadPoolAdapter:
    """Wrapper for Hamilton's FutureAdapter with build-specific defaults.

    Provides a ThreadPool execution adapter with sensible defaults
    for build workloads.

    Parameters
    ----------
    max_workers
        Maximum number of threads. Defaults to min(32, cpu_count + 4).
    thread_name_prefix
        Prefix for thread names.
    result_builder
        Optional ResultBuilder for output aggregation.

    Examples
    --------
    >>> adapter = ThreadPoolAdapter(max_workers=8)
    >>> dr = driver.Builder().with_adapters(adapter).build()
    """

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "hamilton-build",
        result_builder: ResultBuilder | None = None,
    ) -> None:
        """Initialize the threadpool adapter."""
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.result_builder = result_builder
        self._delegate: Any = None

    def _ensure_delegate(self) -> Any:
        """Lazily create the delegate adapter.

        Returns
        -------
        Any
            Underlying Hamilton adapter instance.
        """
        if self._delegate is not None:
            return self._delegate

        from hamilton.plugins.h_threadpool import FutureAdapter

        self._delegate = FutureAdapter(
            max_workers=self.max_workers,
            thread_name_prefix=self.thread_name_prefix,
            result_builder=self.result_builder,
        )
        return self._delegate

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying adapter.

        Returns
        -------
        Any
            Attribute value resolved from the delegate adapter.
        """
        delegate = self._ensure_delegate()
        return getattr(delegate, name)


def create_parallel_adapter(
    backend: str | ExecutionBackend = ExecutionBackend.SEQUENTIAL,
    *,
    max_workers: int | None = None,
    thread_name_prefix: str = "hamilton-build",
    result_builder: ResultBuilder | None = None,
) -> Any | None:
    """Create a parallel execution adapter.

    Factory function for creating execution adapters based on the
    specified backend.

    Parameters
    ----------
    backend
        Execution backend to use. Can be string or ExecutionBackend enum.
    max_workers
        Maximum number of parallel workers.
    thread_name_prefix
        Prefix for thread names (threadpool only).
    result_builder
        Optional ResultBuilder for output aggregation.

    Returns
    -------
    Any | None
        Adapter instance, or None for sequential execution.

    Examples
    --------
    >>> adapter = create_parallel_adapter("threadpool", max_workers=4)
    >>> if adapter:
    ...     dr = driver.Builder().with_adapters(adapter).build()

    >>> # Auto-select backend:
    >>> adapter = create_parallel_adapter("auto")
    """
    # Convert string to enum
    if isinstance(backend, str):
        try:
            backend = ExecutionBackend(backend.lower())
        except ValueError:
            log.warning("Unknown backend %s, using sequential", backend)
            backend = ExecutionBackend.SEQUENTIAL

    # Handle auto selection
    if backend == ExecutionBackend.AUTO:
        # For now, default to threadpool for auto
        # Future: Could check workload characteristics
        backend = ExecutionBackend.THREADPOOL
        log.info("Auto-selected backend: %s", backend.value)

    # Create adapter based on backend
    if backend == ExecutionBackend.SEQUENTIAL:
        # No adapter needed for sequential execution
        return None

    if backend == ExecutionBackend.THREADPOOL:
        return ThreadPoolAdapter(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            result_builder=result_builder,
        )

    # Future backends would be handled here
    log.warning("Backend %s not yet implemented, using sequential", backend.value)
    return None


def create_adapter_from_config(config: ParallelConfig) -> Any | None:
    """Create adapter from ParallelConfig.

    Parameters
    ----------
    config
        Parallel execution configuration.

    Returns
    -------
    Any | None
        Adapter instance, or None for sequential execution.
    """
    return create_parallel_adapter(
        backend=config.backend,
        max_workers=config.max_workers,
        thread_name_prefix=config.thread_name_prefix,
    )
