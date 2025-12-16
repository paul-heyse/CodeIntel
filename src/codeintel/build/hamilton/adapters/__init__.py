"""Hamilton execution adapters for parallel and distributed execution.

This package provides adapters for different execution backends:
- ThreadPool: Multi-threaded execution for I/O-bound workloads
- Async: Native async/await execution (future)
- Ray: Distributed execution (requires ray) (future)
- Dask: Distributed execution (requires dask) (future)

Examples
--------
Using ThreadPool adapter:

>>> from codeintel.build.hamilton.adapters import create_parallel_adapter
>>> adapter = create_parallel_adapter("threadpool", max_workers=4)
>>> dr = driver.Builder().with_adapters(adapter).build()

Using the adapter factory:

>>> adapter = create_parallel_adapter("threadpool")
>>> # Or let it auto-detect:
>>> adapter = create_parallel_adapter("auto")
"""
from __future__ import annotations

from codeintel.build.hamilton.adapters.parallel import (
    ExecutionBackend,
    ParallelConfig,
    create_parallel_adapter,
    get_available_backends,
)

__all__ = [
    "ExecutionBackend",
    "ParallelConfig",
    "create_parallel_adapter",
    "get_available_backends",
]
