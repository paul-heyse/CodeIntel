"""Unified operation execution pipeline.

This module provides the OperationExecutor that orchestrates the complete
lifecycle of CLI operation execution, integrating validation, middleware,
progress tracking, rendering, and resilience patterns.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_middleware import MiddlewareStack, get_middleware_stack
from codeintel.cli.cli_progress import ProgressTracker, get_progress_tracker
from codeintel.cli.cli_render import OutputRenderer, get_renderer, render_cli_result
from codeintel.cli.cli_resilience import RetryPolicy
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.cli_validation import ValidationSchema
from codeintel.cli.resilience_middleware import (
    CircuitBreakerRegistry,
    ResilienceConfig,
    ResilienceMiddleware,
    execute_with_retry,
)
from codeintel.cli.results import CliResult

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
    """

    operation_id: str
    handler: Callable[..., CliResult[T]]
    category: OperationCategory = OperationCategory.READ
    param_schema: ValidationSchema | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = False
    retry_policy: RetryPolicy | None = None
    timeout: float | None = None
    description: str = ""


@dataclass
class ExecutionContext:
    """Context passed through the execution pipeline.

    Parameters
    ----------
    operation_id
        The operation being executed.
    params
        Validated operation parameters.
    output_format
        Requested output format.
    start_time
        Execution start timestamp.
    metadata
        Additional context metadata.
    """

    operation_id: str
    params: dict[str, Any]
    output_format: OutputFormat
    start_time: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed execution time.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        return time.monotonic() - self.start_time


@dataclass
class ExecutionResult[T]:
    """Result of operation execution with metrics.

    Parameters
    ----------
    result
        The CliResult from the handler.
    duration_seconds
        Total execution duration.
    validation_errors
        Any validation errors encountered.
    retries
        Number of retry attempts.
    """

    result: CliResult[T]
    duration_seconds: float
    validation_errors: list[str] = field(default_factory=list)
    retries: int = 0


class OperationExecutor:
    """Orchestrate the complete operation execution pipeline.

    This class integrates validation, middleware, progress tracking,
    rendering, and resilience patterns into a single, consistent execution flow.

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

        # Configure resilience middleware if enabled
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
        registry = CircuitBreakerRegistry(config)
        resilience_mw = ResilienceMiddleware(config=config, breaker_registry=registry)
        self._middleware.add(resilience_mw)

    def execute[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        output_format: OutputFormat = OutputFormat.TEXT,
        render: bool = True,
    ) -> ExecutionResult[T]:
        """Execute an operation through the full pipeline.

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
        ctx = ExecutionContext(
            operation_id=spec.operation_id,
            params=params,
            output_format=output_format,
        )

        LOG.debug(
            "Starting operation execution",
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

        # Phase 2: Execute with middleware and progress
        result = self._execute_with_middleware(spec, ctx)

        # Phase 3: Render output
        if render:
            self._render_result(result, output_format)

        return ExecutionResult(
            result=result,
            duration_seconds=ctx.elapsed_seconds,
        )

    def _validate(
        self,
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
        # Use self to satisfy linter (method needs access to instance for future extensions)
        _ = self
        if spec.param_schema is None:
            return []

        result = spec.param_schema.validate(params)
        if result.is_valid:
            return []

        return [f"{e.field}: {e.message}" for e in result.errors]

    def _create_validation_error_result(
        self,
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
        # Use self to satisfy linter (method needs access to instance for future extensions)
        _ = self
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Validation Failed",
                detail="\n".join(errors),
                status=400,
                extensions={"errors": errors},
            )
        )

    def _execute_with_middleware[T](
        self,
        spec: OperationSpec[T],
        ctx: ExecutionContext,
    ) -> CliResult[T]:
        """Execute handler with middleware and progress.

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
        with self._middleware.wrap(spec.operation_id, ctx.params):
            if spec.requires_progress:
                return self._execute_with_progress(spec, ctx)
            return self._execute_handler(spec, ctx)

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
        # Get retry policy
        retry_policy = spec.retry_policy
        if retry_policy is None and spec.retryable and self._default_retry_policy:
            retry_policy = self._default_retry_policy

        # Execute with retry if policy is available
        if retry_policy is not None:
            return execute_with_retry(
                spec.handler,
                ctx.params,
                retry_policy,
                operation_id=spec.operation_id,
            )
        return spec.handler(**ctx.params)

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


__all__ = [
    "ExecutionContext",
    "ExecutionResult",
    "OperationCategory",
    "OperationExecutor",
    "OperationSpec",
    "configure_executor",
    "get_executor",
]
