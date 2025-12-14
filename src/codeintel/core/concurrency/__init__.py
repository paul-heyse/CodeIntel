"""Unified concurrency utilities.

This module provides worker pool and async utilities.

Examples
--------
Using worker pool context manager:

>>> from codeintel.core.concurrency import worker_pool
>>>
>>> with worker_pool("thread", 4) as executor:
...     futures = [executor.submit(task, arg) for arg in args]

Using executor factory:

>>> from codeintel.core.concurrency import executor_factory
>>>
>>> factory = executor_factory("process", 8)
>>> executor = factory()
"""

from codeintel.core.concurrency.async_utils import (
    run_sync,
)
from codeintel.core.concurrency.workers import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)

__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MIN_WORKERS",
    "WorkerConfig",
    "create_executor",
    "executor_factory",
    "resolve_worker_count",
    "run_sync",
    "worker_pool",
]
