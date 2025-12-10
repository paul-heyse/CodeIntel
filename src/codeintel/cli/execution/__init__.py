"""Unified execution pipeline for CLI operations.

This package provides a single execution infrastructure that supports
handler-based operations with consistent progress tracking.

The canonical `OperationSpec` is defined in `execution/registry.py` and
is used by `@cli_command` decorator and the introspection system.

Public surface
--------------
Use the registry-based APIs (`OperationSpec`, `register_operation`, `execute_operation`) as
the supported entry points for defining and running operations. Legacy executor-style types
such as OperationExecutor/OperationCategory are intentionally not re-exported here; build
on the handler/registry flow instead.

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
    "OperationRegistry",
    "OperationSpec",
    "ProgressConfig",
    "ProgressEvent",
    "ProgressRenderer",
    "ProgressState",
    "ProgressStreamConfig",
    "ProgressTracker",
    "StreamingHandler",
    "StreamingResult",
    "SyncHandler",
    "configure_progress",
    "execute_operation",
    "get_handler_type",
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
