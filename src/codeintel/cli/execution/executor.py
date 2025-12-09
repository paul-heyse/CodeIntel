"""Unified operation executor for CLI operations.

Provide a single executor that handles sync, async, and streaming
handlers with consistent middleware, resilience, and progress tracking.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import types
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_render import (
    OutputRenderer,
    get_renderer,
    render_cli_result,
)
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.cli_validation import ValidationSchema
from codeintel.cli.execution.context import ExecutionContext, ExecutionResult
from codeintel.cli.execution.middleware import (
    MiddlewareStack,
    get_middleware_stack,
)
from codeintel.cli.execution.progress import (
    ProgressTracker,
    get_progress_tracker,
)
from codeintel.cli.execution.types import (
    AnyHandler,
    ProgressEvent,
    ProgressState,
    StreamingResult,
    SyncHandler,
    get_handler_type,
)
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from codeintel.cli.resilience import (
        ResilienceConfig,
        RetryPolicy,
    )


_RESILIENCE_MODULE: types.ModuleType | None = None


def _get_resilience_module() -> types.ModuleType:
    """Lazy import resilience module to avoid circular imports.

    Returns
    -------
    types.ModuleType
        The resilience module.
    """
    global _RESILIENCE_MODULE  # noqa: PLW0603
    if _RESILIENCE_MODULE is None:
        _RESILIENCE_MODULE = importlib.import_module("codeintel.cli.resilience")
    return _RESILIENCE_MODULE

LOG = logging.getLogger(__name__)


class OperationCategory(Enum):
    """Categories of operations for behavior configuration."""

    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    BUILD = "build"


@dataclass(frozen=True)
class OperationSpec[T]:
    """Specification for an operation's execution behavior.

    Support sync handlers, async handlers, and streaming handlers.
    Handler type is auto-detected if not specified.

    Parameters
    ----------
    operation_id
        Unique identifier for the operation.
    handler
        The handler function to execute.
    category
        Operation category for behavior configuration.
    param_schema
        Optional validation schema for parameters.
    requires_progress
        Whether to show progress bar.
    estimated_duration
        Estimated duration in seconds (for progress).
    retryable
        Whether the operation can be retried on failure.
    retry_policy
        Custom retry policy (uses default if retryable=True and not set).
    timeout
        Maximum execution time in seconds.
    description
        Human-readable operation description.
    is_async
        Whether handler is async (auto-detected if None).
    is_streaming
        Whether handler is streaming (auto-detected if None).
    """

    operation_id: str
    handler: AnyHandler[T]
    category: OperationCategory = OperationCategory.READ
    param_schema: ValidationSchema | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = False
    retry_policy: RetryPolicy | None = None
    timeout: float | None = None
    description: str = ""
    is_async: bool | None = None
    is_streaming: bool | None = None

    def __post_init__(self) -> None:
        """Auto-detect handler type if not specified."""
        handler_type = get_handler_type(self.handler)
        if self.is_async is None:
            object.__setattr__(self, "is_async", handler_type in {"async", "streaming"})
        if self.is_streaming is None:
            object.__setattr__(self, "is_streaming", handler_type == "streaming")


class OperationExecutor:
    """Execute operations with middleware, resilience, and progress.

    Handle sync, async, and streaming handlers transparently through
    a single unified interface.

    Parameters
    ----------
    middleware_stack
        Stack of middleware to apply.
    progress_tracker
        Progress tracker for long operations.
    default_renderer
        Default output renderer.
    resilience_config
        Configuration for retry and circuit breaker behavior.
    """

    def __init__(
        self,
        middleware_stack: MiddlewareStack | None = None,
        progress_tracker: ProgressTracker | None = None,
        default_renderer: OutputRenderer | None = None,
        resilience_config: ResilienceConfig | None = None,
    ) -> None:
        """Initialize the executor."""
        self._middleware = middleware_stack or get_middleware_stack()
        self._progress = progress_tracker or get_progress_tracker()
        self._default_renderer = default_renderer
        self._resilience_config = resilience_config
        self._default_retry_policy: RetryPolicy | None = None

        if resilience_config is not None:
            self._configure_resilience(resilience_config)

    def _configure_resilience(self, config: ResilienceConfig) -> None:
        """Configure resilience middleware.

        Parameters
        ----------
        config
            Resilience configuration.
        """
        self._default_retry_policy = config.default_retry_policy
        # Create registry for circuit breaker support
        # Note: The resilience middleware integration will be handled
        # by the existing system until full migration is complete
        resilience = _get_resilience_module()
        _ = resilience.CircuitBreakerRegistry(config)

    def execute[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        output_format: OutputFormat = OutputFormat.TEXT,
        render: bool = True,
    ) -> ExecutionResult[T]:
        """Execute an operation (sync entry point).

        If handler is async, run it in an event loop.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        output_format
            Desired output format.
        render
            Whether to render output (False for programmatic use).

        Returns
        -------
        ExecutionResult[T]
            Execution result with metrics.
        """
        # If handler is async/streaming, delegate to async executor
        if spec.is_async or spec.is_streaming:
            # Check if we're already in an async context
            with contextlib.suppress(RuntimeError):
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Already in async context - use sync handler path
                    # This prevents nested event loop issues
                    return self._execute_sync(spec, params, output_format, render=render)

            # Run async in new event loop
            return asyncio.run(
                self.execute_async(spec, params, output_format=output_format, render=render)
            )

        return self._execute_sync(spec, params, output_format, render=render)

    def _execute_sync[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        output_format: OutputFormat,
        *,
        render: bool,
    ) -> ExecutionResult[T]:
        """Execute a sync handler.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        output_format
            Output format.
        render
            Whether to render output.

        Returns
        -------
        ExecutionResult[T]
            Execution result.
        """
        ctx = ExecutionContext.for_sync(spec.operation_id, params, output_format)

        LOG.debug(
            "Starting sync operation execution",
            extra={"operation_id": spec.operation_id, "params": params},
        )

        # Phase 1: Validation
        validation_errors = self._validate(spec, params)
        if validation_errors:
            result: CliResult[T] = self._create_validation_error_result(validation_errors)
            return ExecutionResult(
                result=result,
                duration_seconds=ctx.elapsed_seconds,
                validation_errors=validation_errors,
            )

        # Phase 2: Execute with middleware
        mw_contexts = self._middleware.execute_before(ctx)

        try:
            if spec.requires_progress:
                result = self._execute_with_progress(spec, ctx)
            else:
                result = self._execute_handler(spec, ctx)

            result = self._middleware.execute_after(ctx, result, mw_contexts)

        except Exception as exc:
            final_exc = self._middleware.execute_on_error(ctx, exc, mw_contexts)
            if final_exc is not None:
                raise final_exc from exc
            result = self._create_error_result(exc)

        # Phase 3: Render output
        if render:
            self._render_result(result, output_format)

        return ExecutionResult(
            result=result,
            duration_seconds=ctx.elapsed_seconds,
        )

    async def execute_async[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        output_format: OutputFormat = OutputFormat.TEXT,
        render: bool = True,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult[T]:
        """Execute an operation asynchronously.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        output_format
            Desired output format.
        render
            Whether to render output.
        timeout_seconds
            Operation timeout in seconds.

        Returns
        -------
        ExecutionResult[T]
            Execution result with metrics.
        """
        ctx = ExecutionContext.for_async(spec.operation_id, params, output_format)
        timeout = timeout_seconds or spec.timeout
        was_cancelled = False

        LOG.debug(
            "Starting async operation execution",
            extra={"operation_id": spec.operation_id, "params": params},
        )

        # Phase 1: Validation
        validation_errors = self._validate(spec, params)
        if validation_errors:
            result: CliResult[T] = self._create_validation_error_result(validation_errors)
            return ExecutionResult(
                result=result,
                duration_seconds=ctx.elapsed_seconds,
                validation_errors=validation_errors,
            )

        # Phase 2: Execute with middleware
        mw_contexts = await self._middleware.execute_before_async(ctx)

        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    self._execute_async_handler(spec, ctx),
                    timeout=timeout,
                )
            else:
                result = await self._execute_async_handler(spec, ctx)

            result = await self._middleware.execute_after_async(ctx, result, mw_contexts)

        except asyncio.CancelledError:
            was_cancelled = True
            result = CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:operation/cancelled",
                    title="Operation Cancelled",
                    detail=f"Operation {spec.operation_id} was cancelled",
                    status=499,
                ),
            )
        except TimeoutError:
            result = CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:operation/timeout",
                    title="Operation Timeout",
                    detail=f"Operation {spec.operation_id} exceeded timeout",
                    status=504,
                ),
            )
        except Exception as exc:
            final_exc = await self._middleware.execute_on_error_async(ctx, exc, mw_contexts)
            if final_exc is not None:
                raise final_exc from exc
            result = self._create_error_result(exc)

        # Phase 3: Render output
        if render:
            self._render_result(result, output_format)

        return ExecutionResult(
            result=result,
            duration_seconds=ctx.elapsed_seconds,
            was_cancelled=was_cancelled,
        )

    async def stream[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamingResult[T]]:
        """Stream execution with progress.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        cancel_event
            Event to signal cancellation.

        Yields
        ------
        StreamingResult[T]
            Progress events and final result.

        Raises
        ------
        asyncio.CancelledError
            If operation is cancelled.
        """
        cancel_event = cancel_event or asyncio.Event()

        # Initial progress event
        yield StreamingResult[T](
            progress=ProgressEvent(
                operation_id=spec.operation_id,
                state=ProgressState.RUNNING,
                message=f"Starting {spec.operation_id}",
            ),
        )

        def check_cancellation() -> None:
            """Check if cancelled.

            Raises
            ------
            asyncio.CancelledError
                If cancellation event is set.
            """
            if cancel_event.is_set():
                raise asyncio.CancelledError

        try:
            check_cancellation()

            result = await self.execute_async(spec, params, render=False)

            state = ProgressState.COMPLETED if result.result.success else ProgressState.FAILED
            yield StreamingResult[T](
                progress=ProgressEvent(
                    operation_id=spec.operation_id,
                    state=state,
                    progress=1.0 if result.result.success else None,
                    message="Completed" if result.result.success else "Failed",
                ),
            )

            yield StreamingResult[T](result=result.result)

        except asyncio.CancelledError:
            yield StreamingResult[T](
                progress=ProgressEvent(
                    operation_id=spec.operation_id,
                    state=ProgressState.CANCELLED,
                    message="Operation cancelled",
                ),
            )
            raise

    @staticmethod
    def _validate(
        spec: OperationSpec[Any],
        params: dict[str, Any],
    ) -> list[str]:
        """Validate operation parameters.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Parameters to validate.

        Returns
        -------
        list[str]
            List of validation error messages.
        """
        if spec.param_schema is None:
            return []

        result = spec.param_schema.validate(params)
        if result.is_valid:
            return []

        return [f"{e.field}: {e.message}" for e in result.errors]

    @staticmethod
    def _create_validation_error_result(
        errors: list[str],
    ) -> CliResult[Any]:
        """Create a CliResult for validation errors.

        Parameters
        ----------
        errors
            Validation error messages.

        Returns
        -------
        CliResult[Any]
            Error result with validation details.
        """
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Validation Failed",
                detail="\n".join(errors),
                status=400,
                extensions={"errors": errors},
            )
        )

    @staticmethod
    def _create_error_result(
        exc: Exception,
    ) -> CliResult[Any]:
        """Create a CliResult for general errors.

        Parameters
        ----------
        exc
            Exception that occurred.

        Returns
        -------
        CliResult[Any]
            Error result.
        """
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:operation/error",
                title="Operation Error",
                detail=str(exc),
                status=500,
            ),
        )

    def _execute_handler[T](
        self,
        spec: OperationSpec[T],
        ctx: ExecutionContext,
    ) -> CliResult[T]:
        """Execute handler with optional retry logic.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        retry_policy = spec.retry_policy
        if retry_policy is None and spec.retryable and self._default_retry_policy:
            retry_policy = self._default_retry_policy

        if retry_policy is not None:
            resilience = _get_resilience_module()
            options = resilience.RetryOptions(operation_id=spec.operation_id)
            return resilience.execute_cli_with_retry(
                spec.handler,  # type: ignore[arg-type]
                ctx.params,
                retry_policy,
                options,
            )
        return spec.handler(**ctx.params)  # type: ignore[return-value]

    async def _execute_async_handler[T](
        self,
        spec: OperationSpec[T],
        ctx: ExecutionContext,
    ) -> CliResult[T]:
        """Execute an async handler.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        handler = spec.handler
        handler_type = get_handler_type(handler)

        if handler_type == "streaming":
            # For streaming handlers, collect final result
            final_result: CliResult[T] | None = None
            async for item in handler(**ctx.params):  # type: ignore[misc]
                if item.is_result and item.result is not None:
                    final_result = item.result
            if final_result is not None:
                return final_result
            return self._create_error_result(
                RuntimeError("Streaming handler did not yield a final result")
            )

        if handler_type == "async":
            return await handler(**ctx.params)  # type: ignore[misc]

        # Sync handler - run in thread pool
        # Cast the handler to sync type since we've verified it's not async/streaming
        sync_handler = cast("SyncHandler[T]", spec.handler)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: sync_handler(**ctx.params),
        )

    def _execute_with_progress[T](
        self,
        spec: OperationSpec[T],
        ctx: ExecutionContext,
    ) -> CliResult[T]:
        """Execute handler with progress tracking.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        with self._progress:
            task_id = self._progress.add_task(
                spec.description or spec.operation_id,
                total=spec.estimated_duration,
            )
            try:
                result = self._execute_handler(spec, ctx)
            except Exception:
                self._progress.update(task_id, description="[red]Failed[/red]")
                raise
            else:
                self._progress.complete(task_id)
                return result

    def _render_result[T](
        self,
        result: CliResult[T],
        output_format: OutputFormat,
    ) -> None:
        """Render the operation result.

        Parameters
        ----------
        result
            Result to render.
        output_format
            Output format.
        """
        renderer = self._default_renderer or get_renderer(output_format)
        render_cli_result(result, renderer)


# Global executor instance
_EXECUTOR: OperationExecutor | None = None


def get_executor() -> OperationExecutor:
    """Get the global operation executor.

    Returns
    -------
    OperationExecutor
        Global executor instance.
    """
    global _EXECUTOR  # noqa: PLW0603
    if _EXECUTOR is None:
        _EXECUTOR = OperationExecutor()
    return _EXECUTOR


def configure_executor(
    *,
    middleware_stack: MiddlewareStack | None = None,
    progress_tracker: ProgressTracker | None = None,
    default_renderer: OutputRenderer | None = None,
    resilience_config: ResilienceConfig | None = None,
) -> OperationExecutor:
    """Configure the global executor.

    Parameters
    ----------
    middleware_stack
        Custom middleware stack.
    progress_tracker
        Custom progress tracker.
    default_renderer
        Custom default renderer.
    resilience_config
        Resilience configuration for retry and circuit breaker.

    Returns
    -------
    OperationExecutor
        Configured executor.
    """
    global _EXECUTOR  # noqa: PLW0603
    _EXECUTOR = OperationExecutor(
        middleware_stack=middleware_stack,
        progress_tracker=progress_tracker,
        default_renderer=default_renderer,
        resilience_config=resilience_config,
    )
    return _EXECUTOR


async def run_async_operation[T](
    spec: OperationSpec[T],
    params: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> CliResult[T]:
    """Run an operation asynchronously.

    Convenience function for running a single async operation.

    Parameters
    ----------
    spec
        Operation specification.
    params
        Operation parameters.
    timeout_seconds
        Optional timeout in seconds.

    Returns
    -------
    CliResult[T]
        Operation result.
    """
    executor = get_executor()
    result = await executor.execute_async(spec, params, timeout_seconds=timeout_seconds)
    return result.result


def run_sync[T](
    spec: OperationSpec[T],
    params: dict[str, Any],
) -> CliResult[T]:
    """Run an async-capable operation synchronously.

    Parameters
    ----------
    spec
        Operation specification.
    params
        Operation parameters.

    Returns
    -------
    CliResult[T]
        Operation result.
    """
    executor = get_executor()
    result = executor.execute(spec, params, render=False)
    return result.result


__all__ = [
    "OperationCategory",
    "OperationExecutor",
    "OperationSpec",
    "configure_executor",
    "get_executor",
    "run_async_operation",
    "run_sync",
]
