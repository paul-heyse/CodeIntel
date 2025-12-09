"""Unified execution pipeline for CLI operations.

This package provides a single execution infrastructure that supports
sync, async, and streaming handlers with consistent middleware,
resilience, and progress tracking.

Examples
--------
Execute a sync operation:

>>> from codeintel.cli.execution import OperationSpec, get_executor
>>> spec = OperationSpec(
...     operation_id="my.operation",
...     handler=my_handler,
... )
>>> result = get_executor().execute(spec, {"arg": "value"})

Execute an async operation:

>>> import asyncio
>>> from codeintel.cli.execution import run_async_operation
>>> result = asyncio.run(run_async_operation(spec, {"arg": "value"}))
"""

from codeintel.cli.execution.context import (
    ExecutionContext,
    ExecutionResult,
)
from codeintel.cli.execution.executor import (
    OperationCategory,
    OperationExecutor,
    OperationSpec,
    configure_executor,
    get_executor,
    run_async_operation,
    run_sync,
)
from codeintel.cli.execution.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewareStack,
    ProgressMiddleware,
    TimingMiddleware,
    TracingMiddleware,
    configure_default_middleware,
    get_middleware_stack,
)
from codeintel.cli.execution.progress import (
    ProgressRenderer,
    ProgressStreamConfig,
    ProgressTracker,
    configure_progress,
    get_progress_tracker,
    iter_with_progress,
    progress_context,
    progress_generator,
    stream_progress,
)
from codeintel.cli.execution.types import (
    AnyHandler,
    AsyncHandler,
    ProgressConfig,
    ProgressEvent,
    ProgressState,
    StreamingHandler,
    StreamingResult,
    SyncHandler,
    get_handler_type,
    is_async_handler,
    is_streaming_handler,
)

__all__ = [
    "AnyHandler",
    "AsyncHandler",
    "ExecutionContext",
    "ExecutionResult",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "Middleware",
    "MiddlewareStack",
    "OperationCategory",
    "OperationExecutor",
    "OperationSpec",
    "ProgressConfig",
    "ProgressEvent",
    "ProgressMiddleware",
    "ProgressRenderer",
    "ProgressState",
    "ProgressStreamConfig",
    "ProgressTracker",
    "StreamingHandler",
    "StreamingResult",
    "SyncHandler",
    "TimingMiddleware",
    "TracingMiddleware",
    "configure_default_middleware",
    "configure_executor",
    "configure_progress",
    "get_executor",
    "get_handler_type",
    "get_middleware_stack",
    "get_progress_tracker",
    "is_async_handler",
    "is_streaming_handler",
    "iter_with_progress",
    "progress_context",
    "progress_generator",
    "run_async_operation",
    "run_sync",
    "stream_progress",
]
