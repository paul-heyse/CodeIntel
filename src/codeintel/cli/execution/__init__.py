"""Unified execution pipeline for CLI operations.

This package provides a single execution infrastructure that supports
handler-based operations with consistent middleware, resilience, and
progress tracking.

The canonical `OperationSpec` is defined in `execution/registry.py` and
is used by `@cli_command` decorator and the introspection system.

Examples
--------
Register an operation:

>>> from codeintel.cli.execution import OperationSpec, register_operation
>>> from codeintel.cli.handlers.context import HandlerContext
>>> from codeintel.cli.core import CliResult
>>>
>>> def my_handler(ctx: HandlerContext) -> CliResult:  # doctest: +SKIP
...     return CliResult.ok({"status": "done"})
>>>
>>> spec = OperationSpec(  # doctest: +SKIP
...     operation_id="my.operation",
...     name="My Operation",
...     description="Does something",
...     handler=my_handler,
...     group="my",
... )
>>> register_operation(spec)  # doctest: +SKIP
"""

from codeintel.cli.execution.context import (
    ExecutionContext,
    ExecutionResult,
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
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    execute_operation,
    get_registry,
    register_operation,
    reset_registry,
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
    "OperationRegistry",
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
    "configure_progress",
    "execute_operation",
    "get_handler_type",
    "get_middleware_stack",
    "get_progress_tracker",
    "get_registry",
    "is_async_handler",
    "is_streaming_handler",
    "iter_with_progress",
    "progress_context",
    "progress_generator",
    "register_operation",
    "reset_registry",
    "stream_progress",
]
