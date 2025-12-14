"""Worker pool utilities.

This module provides utilities for managing worker pools, including
configuration, lifecycle management, and factory functions.

Examples
--------
Using a worker pool context manager:

>>> from codeintel.core.concurrency import worker_pool
>>>
>>> with worker_pool("thread", 4) as executor:
...     futures = [executor.submit(task, arg) for arg in args]

Using executor factory for deferred creation:

>>> from codeintel.core.concurrency import executor_factory
>>>
>>> factory = executor_factory("process", 8)
>>> executor = factory()
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from concurrent.futures import Executor

log = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 16
DEFAULT_MIN_WORKERS = 2


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for worker pools.

    Attributes
    ----------
    max_workers
        Maximum number of workers.
    executor_type
        Type of executor (thread or process).
    env_var
        Optional environment variable name for worker count override.
    default_max
        Maximum worker count when derived from CPU.
    default_min
        Minimum worker count floor.

    Examples
    --------
    >>> config = WorkerConfig(max_workers=4, executor_type="thread")
    >>> config = WorkerConfig(env_var="MY_WORKERS", executor_type="process")
    """

    max_workers: int | None = None
    executor_type: Literal["thread", "process"] = "thread"
    env_var: str | None = None
    default_max: int = DEFAULT_MAX_WORKERS
    default_min: int = DEFAULT_MIN_WORKERS


def resolve_worker_count(
    requested: int | None = None,
    *,
    env_var: str | None = None,
    default_max: int = DEFAULT_MAX_WORKERS,
    default_min: int = DEFAULT_MIN_WORKERS,
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve worker pool size from explicit value, environment, or CPU count.

    Parameters
    ----------
    requested
        Explicit worker count if provided (takes precedence).
    env_var
        Environment variable name to check for override.
    default_max
        Maximum worker count when derived from CPU.
    default_min
        Minimum worker count floor.
    env
        Optional environment mapping to read overrides from.

    Returns
    -------
    int
        Resolved worker count.

    Examples
    --------
    >>> resolve_worker_count(4)
    4
    >>> resolve_worker_count(env_var="MY_WORKERS")  # Uses env or CPU count
    8
    """
    if requested is not None and requested > 0:
        return requested

    environment = env or os.environ

    if env_var:
        env_value = environment.get(env_var)
        if env_value:
            try:
                value = int(env_value)
                if value > 0:
                    return value
            except ValueError:
                log.warning("Ignoring invalid %s=%s", env_var, env_value)

    cpu_count = os.cpu_count() or 1
    return min(default_max, max(default_min, cpu_count // 2))


def create_executor(config: WorkerConfig | None = None) -> Executor:
    """Create an executor based on configuration.

    Parameters
    ----------
    config
        Worker configuration.

    Returns
    -------
    Executor
        Thread or process pool executor.

    Examples
    --------
    >>> config = WorkerConfig(max_workers=4, executor_type="thread")
    >>> executor = create_executor(config)
    """
    if config is None:
        config = WorkerConfig()

    max_workers = resolve_worker_count(
        config.max_workers,
        env_var=config.env_var,
        default_max=config.default_max,
        default_min=config.default_min,
    )

    if config.executor_type == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


@contextmanager
def worker_pool(
    kind: Literal["thread", "process"],
    workers: int,
) -> Iterator[Executor]:
    """Context manager for worker pool lifecycle.

    Automatically shuts down the executor when exiting the context.

    Parameters
    ----------
    kind
        Executor type: "thread" or "process".
    workers
        Maximum worker count.

    Yields
    ------
    Executor
        Configured executor for use within the context.

    Examples
    --------
    >>> with worker_pool("thread", 4) as executor:
    ...     results = list(executor.map(process, items))
    """
    config = WorkerConfig(max_workers=workers, executor_type=kind)
    executor = create_executor(config)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


def executor_factory(
    kind: Literal["thread", "process"],
    workers: int,
) -> Callable[[], Executor]:
    """Create a factory function that produces executors.

    This is useful for deferred executor creation in pipeline execution.

    Parameters
    ----------
    kind
        Executor type: "thread" or "process".
    workers
        Maximum worker count.

    Returns
    -------
    Callable[[], Executor]
        Factory function that creates executors.

    Examples
    --------
    >>> factory = executor_factory("process", 8)
    >>> executor = factory()
    >>> executor.shutdown()
    """

    def _factory() -> Executor:
        config = WorkerConfig(max_workers=workers, executor_type=kind)
        return create_executor(config)

    return _factory


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MIN_WORKERS",
    "WorkerConfig",
    "create_executor",
    "executor_factory",
    "resolve_worker_count",
    "worker_pool",
]
