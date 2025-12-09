# CLI Best-in-Class Implementation Plan (Phase 4)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 3 (Completed)

---

## Executive Summary

Phase 4 represents the **integration and maturation** of the CLI infrastructure built in Phases 2-3. While previous phases focused on creating foundational building blocks (unified types, CliResult pattern, middleware, validation, progress tracking, rendering, resilience), Phase 4 connects these components into a cohesive, production-ready system.

The five priorities address:

1. **Full Integration Layer** — Wire all infrastructure into a unified operation execution pipeline
2. **Consistent Error Semantics** — RFC 9457 Problem Details everywhere with actionable error codes
3. **OpenTelemetry Integration** — Complete observability with traces, metrics, and structured logs
4. **Comprehensive Test Suite** — Charter-compliant tests using the new infrastructure
5. **Dynamic Operation Discovery** — Self-documenting CLI with introspection and dynamic completions

### Why Phase 4 Matters

Without Phase 4, we have excellent components that don't work together:

| Component | Created In | Current State | After Phase 4 |
|-----------|------------|---------------|---------------|
| Validation | Phase 2 | Standalone module | Integrated into every operation |
| Middleware | Phase 2 | Manual wiring | Automatic execution chain |
| Progress | Phase 2 | Opt-in per handler | Automatic for long operations |
| Rendering | Phase 3 | Separate from handlers | Unified output pipeline |
| Resilience | Phase 3 | Decorator-only | Middleware-integrated |
| Test Doubles | Phase 3 | Defined | Used across all CLI tests |

---

## Table of Contents

1. [Phase 4.1: Full Integration Layer](#phase-41-full-integration-layer)
2. [Phase 4.2: Consistent Error Semantics](#phase-42-consistent-error-semantics)
3. [Phase 4.3: OpenTelemetry Integration](#phase-43-opentelemetry-integration)
4. [Phase 4.4: Comprehensive Test Suite](#phase-44-comprehensive-test-suite)
5. [Phase 4.5: Dynamic Operation Discovery](#phase-45-dynamic-operation-discovery)
6. [Implementation Timeline](#implementation-timeline)
7. [Success Metrics](#success-metrics)
8. [Migration Guide](#migration-guide)

---

## Phase 4.1: Full Integration Layer

### Value Proposition

The CLI currently has a fragmented execution model:
- Handlers are called directly by Cyclopts commands
- Middleware must be manually wrapped around each call
- Validation is ad-hoc within handlers
- Progress tracking requires explicit setup
- Rendering is inconsistent between handlers

A unified integration layer provides:
- **Single execution path** — Every operation flows through the same pipeline
- **Automatic cross-cutting concerns** — Logging, metrics, validation happen automatically
- **Consistent behavior** — All operations behave the same way
- **Easy extensibility** — Add new concerns without touching handlers

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI Command Entry                                │
│                    (cyclopts_*.py commands)                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OperationExecutor                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Validate   │→ │  Middleware │→ │   Execute   │→ │   Render    │    │
│  │   Input     │  │    Stack    │  │   Handler   │  │   Output    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│        │               │                │                │              │
│        ▼               ▼                ▼                ▼              │
│  cli_validation   cli_middleware   CliResult[T]    cli_render          │
│                   cli_progress                                          │
│                   cli_resilience                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Create `OperationExecutor` class that orchestrates the full execution pipeline
2. Define `OperationSpec` dataclass that describes operation metadata and behavior
3. Implement automatic validation before handler invocation
4. Wire middleware stack with configurable ordering
5. Integrate progress tracking for operations marked as long-running
6. Route all output through the unified renderer
7. Provide lifecycle hooks for extensibility

### Implementation

#### File: `src/codeintel/cli/executor.py`

```python
"""Unified operation execution pipeline.

This module provides the OperationExecutor that orchestrates the complete
lifecycle of CLI operation execution, integrating validation, middleware,
progress tracking, and rendering.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_middleware import MiddlewareStack, get_middleware_stack
from codeintel.cli.cli_progress import ProgressTracker, get_progress_tracker
from codeintel.cli.cli_render import OutputRenderer, get_renderer
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.cli_validation import ValidationResult, ValidationSchema
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.cli_validation import Validator

LOG = logging.getLogger(__name__)

T = TypeVar("T")


class OperationCategory(Enum):
    """Categories of operations for behavior configuration."""

    READ = "read"  # Fast, read-only operations
    WRITE = "write"  # Mutating operations
    COMPUTE = "compute"  # Long-running computations
    NETWORK = "network"  # External service calls
    BUILD = "build"  # Build system operations


@dataclass(frozen=True)
class OperationSpec(Generic[T]):
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
    timeout
        Maximum execution time in seconds.
    description
        Human-readable operation description.
    """

    operation_id: str
    handler: Callable[..., CliResult[T]]
    category: OperationCategory = OperationCategory.READ
    param_schema: ValidationSchema[dict[str, Any]] | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = False
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
class ExecutionResult(Generic[T]):
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
    """Orchestrates the complete operation execution pipeline.

    This class integrates validation, middleware, progress tracking,
    and rendering into a single, consistent execution flow.

    Parameters
    ----------
    middleware_stack
        Stack of middleware to apply.
    progress_tracker
        Progress tracker for long operations.
    default_renderer
        Default output renderer.
    """

    def __init__(
        self,
        middleware_stack: MiddlewareStack | None = None,
        progress_tracker: ProgressTracker | None = None,
        default_renderer: OutputRenderer | None = None,
    ) -> None:
        """Initialize the executor."""
        self._middleware = middleware_stack or get_middleware_stack()
        self._progress = progress_tracker or get_progress_tracker()
        self._default_renderer = default_renderer

    def execute(
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
            result = self._create_validation_error_result(validation_errors)
            return ExecutionResult(
                result=result,
                duration_seconds=ctx.elapsed_seconds,
                validation_errors=validation_errors,
            )

        # Phase 2: Execute with middleware and progress
        result = self._execute_with_middleware(spec, ctx)

        # Phase 3: Render output
        if render:
            renderer = self._default_renderer or get_renderer(output_format)
            self._render_result(result, renderer, spec)

        return ExecutionResult(
            result=result,
            duration_seconds=ctx.elapsed_seconds,
        )

    def _validate(
        self,
        spec: OperationSpec[T],
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
        return CliResult.error(
            ProblemDetail(
                type="urn:codeintel:cli:validation-error",
                title="Validation Failed",
                detail="\n".join(errors),
                status=400,
                extensions={"errors": errors},
            )
        )

    def _execute_with_middleware(
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
            return spec.handler(**ctx.params)

    def _execute_with_progress(
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
                result = spec.handler(**ctx.params)
                self._progress.complete(task_id)
                return result
            except Exception:
                self._progress.update(task_id, description="[red]Failed[/red]")
                raise

    def _render_result(
        self,
        result: CliResult[T],
        renderer: OutputRenderer,
        spec: OperationSpec[T],
    ) -> None:
        """Render the operation result.

        Parameters
        ----------
        result
            Result to render.
        renderer
            Output renderer.
        spec
            Operation specification.
        """
        if not result.success and result.error:
            renderer.render_error(result.error)
        elif result.data is not None:
            renderer.render_object(result.data)


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
```

#### File: `src/codeintel/cli/operation_registry.py`

```python
"""Registry for operation specifications.

Provides a central registry where operations can be registered with their
specifications, enabling dynamic discovery and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from codeintel.cli.executor import OperationCategory, OperationSpec

LOG = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class OperationRegistry:
    """Central registry for operation specifications.

    Parameters
    ----------
    operations
        Mapping of operation IDs to specifications.
    """

    operations: dict[str, OperationSpec[Any]] = field(default_factory=dict)

    def register(self, spec: OperationSpec[T]) -> OperationSpec[T]:
        """Register an operation specification.

        Parameters
        ----------
        spec
            Operation specification to register.

        Returns
        -------
        OperationSpec[T]
            The registered specification.

        Raises
        ------
        ValueError
            If operation ID is already registered.
        """
        if spec.operation_id in self.operations:
            msg = f"Operation already registered: {spec.operation_id}"
            raise ValueError(msg)

        self.operations[spec.operation_id] = spec
        LOG.debug("Registered operation: %s", spec.operation_id)
        return spec

    def get(self, operation_id: str) -> OperationSpec[Any] | None:
        """Get an operation specification by ID.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec[Any] | None
            Specification or None if not found.
        """
        return self.operations.get(operation_id)

    def list_operations(
        self,
        *,
        category: OperationCategory | None = None,
    ) -> list[OperationSpec[Any]]:
        """List registered operations.

        Parameters
        ----------
        category
            Optional category filter.

        Returns
        -------
        list[OperationSpec[Any]]
            Matching operations.
        """
        ops = list(self.operations.values())
        if category is not None:
            ops = [op for op in ops if op.category == category]
        return ops

    def unregister(self, operation_id: str) -> bool:
        """Unregister an operation.

        Parameters
        ----------
        operation_id
            Operation to unregister.

        Returns
        -------
        bool
            True if operation was removed.
        """
        if operation_id in self.operations:
            del self.operations[operation_id]
            return True
        return False


# Global registry instance
_REGISTRY = OperationRegistry()


def get_operation_registry() -> OperationRegistry:
    """Get the global operation registry.

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    return _REGISTRY


def register_operation(spec: OperationSpec[T]) -> OperationSpec[T]:
    """Register an operation with the global registry.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    OperationSpec[T]
        Registered specification.
    """
    return _REGISTRY.register(spec)


__all__ = [
    "OperationRegistry",
    "get_operation_registry",
    "register_operation",
]
```

### Integration with Existing Commands

Update `cyclopts_ops.py` to use the executor:

```python
from codeintel.cli.executor import (
    OperationCategory,
    OperationSpec,
    get_executor,
)
from codeintel.cli.operation_registry import register_operation

# Register operation specifications at module load
OP_LIST_SPEC = register_operation(
    OperationSpec(
        operation_id="op.list",
        handler=op_list_handler_structured,
        category=OperationCategory.READ,
        description="List available operations",
    )
)

OP_CALL_SPEC = register_operation(
    OperationSpec(
        operation_id="op.call",
        handler=op_call_handler_structured,
        category=OperationCategory.COMPUTE,
        requires_progress=True,
        retryable=True,
        description="Call an operation",
    )
)


@op_app.command()
def list_ops(
    output_format: Annotated[OutputFormat, ...] = OutputFormat.TEXT,
) -> None:
    """List available operations."""
    executor = get_executor()
    result = executor.execute(OP_LIST_SPEC, {}, output_format=output_format)
    if not result.result.success:
        raise SystemExit(1)
```

---

## Phase 4.2: Consistent Error Semantics

### Value Proposition

Current error handling is inconsistent:
- Some handlers raise exceptions
- Some return `CliResult.error()`
- Error messages vary in quality
- No standard error codes for automation
- Stack traces leak in some cases

Consistent error semantics provide:
- **Machine-readable errors** — Standard codes enable automation
- **Actionable messages** — Users know how to fix issues
- **Debug support** — `--debug` flag exposes details
- **Error correlation** — Errors can be tracked across systems

### Error Taxonomy

```
urn:codeintel:cli:
├── validation-error           # Input validation failed
│   ├── missing-required       # Required parameter missing
│   ├── invalid-type           # Wrong parameter type
│   ├── invalid-format         # Value doesn't match pattern
│   └── out-of-range           # Value outside allowed range
├── operation-error            # Operation execution failed
│   ├── not-found              # Operation/resource not found
│   ├── timeout                # Operation timed out
│   ├── dependency-failed      # Prerequisite failed
│   └── internal-error         # Unexpected error
├── storage-error              # Storage layer error
│   ├── connection-failed      # Cannot connect to database
│   ├── query-failed           # Query execution failed
│   └── schema-mismatch        # Schema version mismatch
├── config-error               # Configuration error
│   ├── file-not-found         # Config file missing
│   ├── parse-error            # Config file malformed
│   └── invalid-value          # Config value invalid
└── service-error              # External service error
    ├── unavailable            # Service not responding
    ├── rate-limited           # Too many requests
    └── authentication-failed  # Auth error
```

### Implementation

#### File: `src/codeintel/cli/errors.py`

```python
"""Standardized error types and codes for CLI operations.

This module defines the error taxonomy and provides factory functions
for creating RFC 9457 Problem Details with consistent structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from codeintel.cli.cli_errors import ProblemDetail


class ErrorCategory(Enum):
    """Top-level error categories."""

    VALIDATION = "validation-error"
    OPERATION = "operation-error"
    STORAGE = "storage-error"
    CONFIG = "config-error"
    SERVICE = "service-error"


class ValidationErrorCode(Enum):
    """Validation error codes."""

    MISSING_REQUIRED = "missing-required"
    INVALID_TYPE = "invalid-type"
    INVALID_FORMAT = "invalid-format"
    OUT_OF_RANGE = "out-of-range"


class OperationErrorCode(Enum):
    """Operation error codes."""

    NOT_FOUND = "not-found"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILED = "dependency-failed"
    INTERNAL_ERROR = "internal-error"


class StorageErrorCode(Enum):
    """Storage error codes."""

    CONNECTION_FAILED = "connection-failed"
    QUERY_FAILED = "query-failed"
    SCHEMA_MISMATCH = "schema-mismatch"


class ConfigErrorCode(Enum):
    """Configuration error codes."""

    FILE_NOT_FOUND = "file-not-found"
    PARSE_ERROR = "parse-error"
    INVALID_VALUE = "invalid-value"


class ServiceErrorCode(Enum):
    """External service error codes."""

    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate-limited"
    AUTHENTICATION_FAILED = "authentication-failed"


# HTTP status code mappings
_STATUS_CODES: dict[str, int] = {
    ValidationErrorCode.MISSING_REQUIRED.value: 400,
    ValidationErrorCode.INVALID_TYPE.value: 400,
    ValidationErrorCode.INVALID_FORMAT.value: 400,
    ValidationErrorCode.OUT_OF_RANGE.value: 400,
    OperationErrorCode.NOT_FOUND.value: 404,
    OperationErrorCode.TIMEOUT.value: 504,
    OperationErrorCode.DEPENDENCY_FAILED.value: 424,
    OperationErrorCode.INTERNAL_ERROR.value: 500,
    StorageErrorCode.CONNECTION_FAILED.value: 503,
    StorageErrorCode.QUERY_FAILED.value: 500,
    StorageErrorCode.SCHEMA_MISMATCH.value: 500,
    ConfigErrorCode.FILE_NOT_FOUND.value: 404,
    ConfigErrorCode.PARSE_ERROR.value: 400,
    ConfigErrorCode.INVALID_VALUE.value: 400,
    ServiceErrorCode.UNAVAILABLE.value: 503,
    ServiceErrorCode.RATE_LIMITED.value: 429,
    ServiceErrorCode.AUTHENTICATION_FAILED.value: 401,
}


def make_error_type(category: ErrorCategory, code: str) -> str:
    """Create a fully-qualified error type URI.

    Parameters
    ----------
    category
        Error category.
    code
        Specific error code.

    Returns
    -------
    str
        Error type URI.
    """
    return f"urn:codeintel:cli:{category.value}:{code}"


def validation_error(
    code: ValidationErrorCode,
    field: str,
    message: str,
    *,
    value: Any = None,
    suggestion: str | None = None,
) -> ProblemDetail:
    """Create a validation error.

    Parameters
    ----------
    code
        Validation error code.
    field
        Field that failed validation.
    message
        Error message.
    value
        The invalid value (if safe to include).
    suggestion
        Suggestion for fixing the error.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"field": field}
    if value is not None:
        extensions["value"] = str(value)[:100]  # Truncate for safety
    if suggestion:
        extensions["suggestion"] = suggestion

    return ProblemDetail(
        type=make_error_type(ErrorCategory.VALIDATION, code.value),
        title="Validation Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 400),
        extensions=extensions,
    )


def operation_error(
    code: OperationErrorCode,
    operation_id: str,
    message: str,
    *,
    cause: Exception | None = None,
    debug_info: dict[str, Any] | None = None,
) -> ProblemDetail:
    """Create an operation error.

    Parameters
    ----------
    code
        Operation error code.
    operation_id
        The operation that failed.
    message
        Error message.
    cause
        Underlying exception (included in debug mode).
    debug_info
        Additional debug information.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"operation_id": operation_id}
    if cause is not None:
        extensions["cause_type"] = type(cause).__name__
    if debug_info:
        extensions["debug"] = debug_info

    return ProblemDetail(
        type=make_error_type(ErrorCategory.OPERATION, code.value),
        title="Operation Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 500),
        instance=f"/operations/{operation_id}",
        extensions=extensions,
    )


def storage_error(
    code: StorageErrorCode,
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> ProblemDetail:
    """Create a storage error.

    Parameters
    ----------
    code
        Storage error code.
    message
        Error message.
    query
        The failing query (truncated).
    table
        The table involved.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {}
    if query:
        extensions["query"] = query[:200]  # Truncate
    if table:
        extensions["table"] = table

    return ProblemDetail(
        type=make_error_type(ErrorCategory.STORAGE, code.value),
        title="Storage Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 500),
        extensions=extensions,
    )


def config_error(
    code: ConfigErrorCode,
    message: str,
    *,
    path: str | None = None,
    key: str | None = None,
) -> ProblemDetail:
    """Create a configuration error.

    Parameters
    ----------
    code
        Config error code.
    message
        Error message.
    path
        Config file path.
    key
        Config key that failed.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {}
    if path:
        extensions["path"] = path
    if key:
        extensions["key"] = key

    return ProblemDetail(
        type=make_error_type(ErrorCategory.CONFIG, code.value),
        title="Configuration Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 400),
        extensions=extensions,
    )


def service_error(
    code: ServiceErrorCode,
    service: str,
    message: str,
    *,
    retry_after: float | None = None,
) -> ProblemDetail:
    """Create a service error.

    Parameters
    ----------
    code
        Service error code.
    service
        Name of the failing service.
    message
        Error message.
    retry_after
        Seconds to wait before retry.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"service": service}
    if retry_after is not None:
        extensions["retry_after_seconds"] = retry_after

    return ProblemDetail(
        type=make_error_type(ErrorCategory.SERVICE, code.value),
        title="Service Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 503),
        extensions=extensions,
    )


@dataclass
class ErrorContext:
    """Context for error creation with debug support.

    Parameters
    ----------
    debug_mode
        Whether to include debug information.
    correlation_id
        Request correlation ID.
    """

    debug_mode: bool = False
    correlation_id: str | None = None

    def wrap_exception(
        self,
        exc: Exception,
        *,
        operation_id: str | None = None,
    ) -> ProblemDetail:
        """Wrap an exception as a ProblemDetail.

        Parameters
        ----------
        exc
            Exception to wrap.
        operation_id
            Optional operation context.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        debug_info: dict[str, Any] | None = None
        if self.debug_mode:
            import traceback

            debug_info = {
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }

        extensions: dict[str, Any] = {}
        if self.correlation_id:
            extensions["correlation_id"] = self.correlation_id
        if debug_info:
            extensions["debug"] = debug_info

        return ProblemDetail(
            type=make_error_type(ErrorCategory.OPERATION, OperationErrorCode.INTERNAL_ERROR.value),
            title="Internal Error",
            detail=str(exc) if self.debug_mode else "An unexpected error occurred",
            status=500,
            instance=f"/operations/{operation_id}" if operation_id else None,
            extensions=extensions if extensions else None,
        )


__all__ = [
    "ConfigErrorCode",
    "ErrorCategory",
    "ErrorContext",
    "OperationErrorCode",
    "ServiceErrorCode",
    "StorageErrorCode",
    "ValidationErrorCode",
    "config_error",
    "make_error_type",
    "operation_error",
    "service_error",
    "storage_error",
    "validation_error",
]
```

### Debug Mode Integration

Add to `cyclopts_app.py`:

```python
# Global debug flag
_DEBUG_MODE = False


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG_MODE  # noqa: PLW0603
    _DEBUG_MODE = enabled


@app.command()
def main(
    debug: Annotated[
        bool,
        Parameter(help="Enable debug mode with stack traces"),
    ] = False,
) -> None:
    """CodeIntel CLI."""
    set_debug_mode(debug)
```

---

## Phase 4.3: OpenTelemetry Integration

### Value Proposition

Without observability, we're blind to:
- Which operations are slow
- Where errors occur
- How the CLI is being used
- Performance regressions over time

OpenTelemetry integration provides:
- **Distributed tracing** — Track operations across systems
- **Metrics** — Latency percentiles, error rates, throughput
- **Structured logging** — Logs correlated with traces
- **Vendor neutrality** — Export to any observability backend

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Operation                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│     Traces      │ │     Metrics     │ │      Logs       │
│  (Spans + ctx)  │ │  (Counters,     │ │  (Structured,   │
│                 │ │   Histograms)   │ │   correlated)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌──────────────────────────┐
              │    OTEL Collector /      │
              │    Console Exporter      │
              └──────────────────────────┘
```

### Implementation

#### File: `src/codeintel/cli/telemetry.py`

```python
"""OpenTelemetry integration for CLI observability.

Provides tracing, metrics, and structured logging with automatic
correlation and context propagation.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

LOG = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry.

    Parameters
    ----------
    enabled
        Whether telemetry is enabled.
    service_name
        Service name for traces.
    export_traces
        Whether to export traces.
    export_metrics
        Whether to export metrics.
    console_export
        Export to console (for debugging).
    otlp_endpoint
        OTLP collector endpoint.
    """

    enabled: bool = True
    service_name: str = "codeintel-cli"
    export_traces: bool = True
    export_metrics: bool = True
    console_export: bool = False
    otlp_endpoint: str | None = None

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        """Create config from environment variables.

        Returns
        -------
        TelemetryConfig
            Configuration from environment.
        """
        return cls(
            enabled=os.environ.get("OTEL_SDK_DISABLED", "").lower() != "true",
            service_name=os.environ.get("OTEL_SERVICE_NAME", "codeintel-cli"),
            export_traces=os.environ.get("CODEINTEL_EXPORT_TRACES", "true").lower() == "true",
            export_metrics=os.environ.get("CODEINTEL_EXPORT_METRICS", "true").lower() == "true",
            console_export=os.environ.get("CODEINTEL_CONSOLE_TELEMETRY", "false").lower() == "true",
            otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        )


class TelemetryProvider:
    """Provider for OpenTelemetry instrumentation.

    Manages tracer and meter instances, providing a facade
    that gracefully degrades when OTEL is not available.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        """Initialize the telemetry provider.

        Parameters
        ----------
        config
            Telemetry configuration.
        """
        self._config = config or TelemetryConfig.from_env()
        self._tracer: Tracer | None = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize OpenTelemetry if available."""
        if self._initialized or not self._config.enabled:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            resource = Resource.create({"service.name": self._config.service_name})
            provider = TracerProvider(resource=resource)

            if self._config.console_export:
                from opentelemetry.sdk.trace.export import (
                    ConsoleSpanExporter,
                    SimpleSpanProcessor,
                )

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            if self._config.otlp_endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=self._config.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
            self._initialized = True
            LOG.debug("OpenTelemetry initialized")

        except ImportError:
            LOG.debug("OpenTelemetry not available, telemetry disabled")
            self._config.enabled = False

    @property
    def tracer(self) -> Tracer | None:
        """Get the tracer instance.

        Returns
        -------
        Tracer | None
            Tracer or None if not available.
        """
        self._initialize()
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span | None]:
        """Create a trace span.

        Parameters
        ----------
        name
            Span name.
        attributes
            Span attributes.

        Yields
        ------
        Span | None
            Active span or None if tracing disabled.
        """
        if not self._config.enabled or self.tracer is None:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span


@dataclass
class OperationMetrics:
    """Metrics collector for CLI operations.

    Parameters
    ----------
    operation_counts
        Count of operations by ID and status.
    operation_durations
        Duration histograms by operation ID.
    """

    operation_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    operation_durations: dict[str, list[float]] = field(default_factory=dict)

    def record_operation(
        self,
        operation_id: str,
        *,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Record operation execution.

        Parameters
        ----------
        operation_id
            Operation identifier.
        success
            Whether operation succeeded.
        duration_seconds
            Execution duration.
        """
        status = "success" if success else "error"

        if operation_id not in self.operation_counts:
            self.operation_counts[operation_id] = {"success": 0, "error": 0}
        self.operation_counts[operation_id][status] += 1

        if operation_id not in self.operation_durations:
            self.operation_durations[operation_id] = []
        self.operation_durations[operation_id].append(duration_seconds)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary.

        Returns
        -------
        dict[str, Any]
            Metrics summary.
        """
        summary: dict[str, Any] = {}
        for op_id, counts in self.operation_counts.items():
            durations = self.operation_durations.get(op_id, [])
            summary[op_id] = {
                "total_calls": sum(counts.values()),
                "success_count": counts.get("success", 0),
                "error_count": counts.get("error", 0),
                "avg_duration_ms": (sum(durations) / len(durations) * 1000) if durations else 0,
                "p95_duration_ms": self._percentile(durations, 95) * 1000 if durations else 0,
            }
        return summary

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile.

        Parameters
        ----------
        values
            List of values.
        percentile
            Percentile to calculate (0-100).

        Returns
        -------
        float
            Percentile value.
        """
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class TracingMiddleware:
    """Middleware that adds tracing to operations.

    Parameters
    ----------
    provider
        Telemetry provider.
    metrics
        Metrics collector.
    """

    def __init__(
        self,
        provider: TelemetryProvider | None = None,
        metrics: OperationMetrics | None = None,
    ) -> None:
        """Initialize tracing middleware."""
        self._provider = provider or get_telemetry_provider()
        self._metrics = metrics or OperationMetrics()

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Start trace span before operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context with span and start time.
        """
        tracer = self._provider.tracer
        span = None

        if tracer is not None:
            span = tracer.start_span(
                f"cli.operation.{op_id}",
                attributes={
                    "cli.operation_id": op_id,
                    "cli.param_count": len(params),
                },
            )

        return {
            "span": span,
            "start_time": time.monotonic(),
            "op_id": op_id,
        }

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Complete trace span after operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        duration = time.monotonic() - context["start_time"]
        self._metrics.record_operation(op_id, success=True, duration_seconds=duration)

        span = context.get("span")
        if span is not None:
            span.set_attribute("cli.success", True)
            span.set_attribute("cli.duration_ms", duration * 1000)
            span.end()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error in trace span.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        duration = time.monotonic() - context["start_time"]
        self._metrics.record_operation(op_id, success=False, duration_seconds=duration)

        span = context.get("span")
        if span is not None:
            span.set_attribute("cli.success", False)
            span.set_attribute("cli.error_type", type(exc).__name__)
            span.record_exception(exc)
            span.end()


# Global instances
_TELEMETRY_PROVIDER: TelemetryProvider | None = None
_OPERATION_METRICS: OperationMetrics | None = None


def get_telemetry_provider() -> TelemetryProvider:
    """Get the global telemetry provider.

    Returns
    -------
    TelemetryProvider
        Global provider instance.
    """
    global _TELEMETRY_PROVIDER  # noqa: PLW0603
    if _TELEMETRY_PROVIDER is None:
        _TELEMETRY_PROVIDER = TelemetryProvider()
    return _TELEMETRY_PROVIDER


def get_operation_metrics() -> OperationMetrics:
    """Get the global operation metrics.

    Returns
    -------
    OperationMetrics
        Global metrics instance.
    """
    global _OPERATION_METRICS  # noqa: PLW0603
    if _OPERATION_METRICS is None:
        _OPERATION_METRICS = OperationMetrics()
    return _OPERATION_METRICS


__all__ = [
    "OperationMetrics",
    "TelemetryConfig",
    "TelemetryProvider",
    "TracingMiddleware",
    "get_operation_metrics",
    "get_telemetry_provider",
]
```

### Structured Logging Integration

Update logging configuration to include trace context:

```python
# In cli startup
import logging
from codeintel.cli.telemetry import get_telemetry_provider


class TraceContextFilter(logging.Filter):
    """Add trace context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace_id and span_id to record."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            record.trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else None
            record.span_id = format(ctx.span_id, "016x") if ctx.span_id else None
        except (ImportError, Exception):
            record.trace_id = None
            record.span_id = None
        return True
```

---

## Phase 4.4: Comprehensive Test Suite

### Value Proposition

Current CLI tests have gaps:
- Test doubles exist but aren't used
- Many tests use forbidden `monkeypatch`
- No contract tests for `CliResult`
- No golden file tests for output
- Coverage is incomplete

A comprehensive test suite provides:
- **Charter compliance** — No monkeypatch, proper dependency injection
- **Confidence** — High coverage with meaningful tests
- **Regression prevention** — Golden files catch output changes
- **Documentation** — Tests serve as usage examples

### Test Organization

```
tests/cli/
├── _doubles/              # Test doubles (existing)
│   ├── __init__.py
│   └── contexts.py
├── _fixtures/             # Shared fixtures
│   ├── __init__.py
│   ├── operations.py      # Operation test fixtures
│   └── golden/            # Golden files for output testing
│       ├── op_list_text.txt
│       ├── op_list_json.json
│       └── error_validation.json
├── unit/                  # Unit tests for individual components
│   ├── test_executor.py
│   ├── test_validation.py
│   ├── test_errors.py
│   └── test_render.py
├── integration/           # Integration tests for command flows
│   ├── test_op_commands.py
│   ├── test_dataset_commands.py
│   └── test_build_commands.py
├── contract/              # Contract tests for CliResult
│   └── test_cli_result_contract.py
└── conftest.py            # Shared fixtures and configuration
```

### Implementation

#### File: `tests/cli/conftest.py`

```python
"""Shared fixtures for CLI testing.

These fixtures provide consistent test setup using the
test doubles defined in _doubles/ rather than monkeypatching.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.cli._doubles import (
    FakeConsole,
    FakeFileSystem,
    FakeOperationCatalog,
    FakeStorageGateway,
)
from tests.cli._doubles.contexts import CliTestContext, CliTestContextBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def cli_context() -> CliTestContext:
    """Provide an empty CLI test context.

    Returns
    -------
    CliTestContext
        Empty test context.
    """
    return CliTestContext()


@pytest.fixture
def cli_context_builder() -> CliTestContextBuilder:
    """Provide a CLI test context builder.

    Returns
    -------
    CliTestContextBuilder
        Builder for constructing test contexts.
    """
    return CliTestContextBuilder()


@pytest.fixture
def fake_storage() -> FakeStorageGateway:
    """Provide a fake storage gateway.

    Returns
    -------
    FakeStorageGateway
        Empty fake storage.
    """
    return FakeStorageGateway()


@pytest.fixture
def fake_operations() -> FakeOperationCatalog:
    """Provide a fake operation catalog.

    Returns
    -------
    FakeOperationCatalog
        Empty fake catalog.
    """
    return FakeOperationCatalog()


@pytest.fixture
def fake_console() -> FakeConsole:
    """Provide a fake console for output capture.

    Returns
    -------
    FakeConsole
        Fake console with output capture.
    """
    return FakeConsole()


@pytest.fixture
def fake_filesystem(tmp_path: Path) -> FakeFileSystem:
    """Provide a fake filesystem rooted at tmp_path.

    Parameters
    ----------
    tmp_path
        Pytest temporary path.

    Returns
    -------
    FakeFileSystem
        Fake filesystem.
    """
    fs = FakeFileSystem()
    fs.directories.add(str(tmp_path))
    return fs


@pytest.fixture
def golden_path() -> Path:
    """Provide path to golden files directory.

    Returns
    -------
    Path
        Golden files directory.
    """
    return Path(__file__).parent / "_fixtures" / "golden"


@pytest.fixture
def sample_operations() -> list[dict[str, Any]]:
    """Provide sample operation data for testing.

    Returns
    -------
    list[dict[str, Any]]
        Sample operations.
    """
    return [
        {
            "id": "analyze.functions",
            "summary": "Analyze function metrics",
            "tags": ["analytics", "functions"],
        },
        {
            "id": "analyze.coverage",
            "summary": "Analyze test coverage",
            "tags": ["analytics", "testing"],
        },
        {
            "id": "build.index",
            "summary": "Build search index",
            "tags": ["build"],
        },
    ]


@pytest.fixture
def populated_context(
    cli_context_builder: CliTestContextBuilder,
    sample_operations: list[dict[str, Any]],
) -> CliTestContext:
    """Provide a CLI context populated with sample data.

    Parameters
    ----------
    cli_context_builder
        Builder fixture.
    sample_operations
        Sample operations fixture.

    Returns
    -------
    CliTestContext
        Populated test context.
    """
    builder = cli_context_builder
    for op in sample_operations:
        builder = builder.with_operation(
            op["id"],
            summary=op["summary"],
            tags=op["tags"],
        )
    return builder.build()
```

#### File: `tests/cli/unit/test_executor.py`

```python
"""Unit tests for OperationExecutor."""

from __future__ import annotations

from typing import Any

import pytest

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_validation import StringValidator, ValidationSchema
from codeintel.cli.executor import (
    ExecutionResult,
    OperationCategory,
    OperationExecutor,
    OperationSpec,
)
from codeintel.cli.results import CliResult


def _success_handler(**kwargs: Any) -> CliResult[dict[str, Any]]:
    """Test handler that succeeds."""
    return CliResult.ok({"params": kwargs})


def _error_handler(**kwargs: Any) -> CliResult[dict[str, Any]]:
    """Test handler that returns an error."""
    return CliResult.error(
        ProblemDetail(
            type="urn:test:error",
            title="Test Error",
            detail="Handler error",
            status=500,
        )
    )


def _raising_handler(**kwargs: Any) -> CliResult[dict[str, Any]]:
    """Test handler that raises an exception."""
    msg = "Handler exception"
    raise RuntimeError(msg)


class TestOperationExecutor:
    """Tests for OperationExecutor."""

    def test_execute_success(self) -> None:
        """Test successful operation execution."""
        spec = OperationSpec(
            operation_id="test.success",
            handler=_success_handler,
            category=OperationCategory.READ,
        )
        executor = OperationExecutor()

        result = executor.execute(spec, {"key": "value"}, render=False)

        assert result.result.success
        assert result.result.data == {"params": {"key": "value"}}
        assert result.duration_seconds > 0

    def test_execute_handler_error(self) -> None:
        """Test operation that returns error result."""
        spec = OperationSpec(
            operation_id="test.error",
            handler=_error_handler,
            category=OperationCategory.READ,
        )
        executor = OperationExecutor()

        result = executor.execute(spec, {}, render=False)

        assert not result.result.success
        assert result.result.error is not None
        assert result.result.error.title == "Test Error"

    def test_execute_with_validation(self) -> None:
        """Test operation with parameter validation."""
        schema: ValidationSchema[dict[str, Any]] = ValidationSchema()
        schema.add("name", StringValidator(min_length=1))

        spec = OperationSpec(
            operation_id="test.validated",
            handler=_success_handler,
            category=OperationCategory.READ,
            param_schema=schema,
        )
        executor = OperationExecutor()

        # Valid params
        result = executor.execute(spec, {"name": "test"}, render=False)
        assert result.result.success

    def test_execute_validation_failure(self) -> None:
        """Test operation with failing validation."""
        schema: ValidationSchema[dict[str, Any]] = ValidationSchema()
        schema.add("name", StringValidator(min_length=1))

        spec = OperationSpec(
            operation_id="test.validated",
            handler=_success_handler,
            category=OperationCategory.READ,
            param_schema=schema,
        )
        executor = OperationExecutor()

        # Missing required param
        result = executor.execute(spec, {}, render=False)
        assert not result.result.success
        assert len(result.validation_errors) > 0

    def test_execute_category_compute_has_progress(self) -> None:
        """Test that compute operations can have progress."""
        spec = OperationSpec(
            operation_id="test.compute",
            handler=_success_handler,
            category=OperationCategory.COMPUTE,
            requires_progress=True,
            estimated_duration=5.0,
        )
        executor = OperationExecutor()

        result = executor.execute(spec, {}, render=False)
        assert result.result.success
```

#### File: `tests/cli/contract/test_cli_result_contract.py`

```python
"""Contract tests for CliResult.

These tests ensure CliResult maintains its contract across changes:
- Serialization format is stable
- Error structure matches RFC 9457
- Success/failure semantics are consistent
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.results import CliResult


class TestCliResultContract:
    """Contract tests for CliResult."""

    def test_success_result_has_data(self) -> None:
        """Success results must have data attribute."""
        result = CliResult.ok({"key": "value"})

        assert result.success is True
        assert result.data is not None
        assert result.error is None

    def test_error_result_has_problem_detail(self) -> None:
        """Error results must have RFC 9457 Problem Detail."""
        error = ProblemDetail(
            type="urn:test:error",
            title="Test Error",
            detail="Details",
            status=400,
        )
        result = CliResult.error(error)

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.type == "urn:test:error"

    def test_json_serialization_success(self) -> None:
        """Success result JSON has required fields."""
        result = CliResult.ok({"count": 42})
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert "success" in parsed
        assert parsed["success"] is True
        assert "data" in parsed
        assert parsed["data"]["count"] == 42

    def test_json_serialization_error(self) -> None:
        """Error result JSON follows RFC 9457 structure."""
        error = ProblemDetail(
            type="urn:codeintel:cli:validation-error",
            title="Validation Failed",
            detail="Field 'name' is required",
            status=400,
            instance="/operations/test",
            extensions={"field": "name"},
        )
        result = CliResult.error(error)
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is False
        assert "error" in parsed
        err = parsed["error"]

        # RFC 9457 required fields
        assert "type" in err
        assert "title" in err

        # RFC 9457 optional fields
        assert "detail" in err
        assert "status" in err
        assert "instance" in err

    def test_warnings_preserved(self) -> None:
        """Warnings are preserved in result."""
        result = CliResult.ok({"data": 1}, warnings=["Warning 1", "Warning 2"])

        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings

    def test_metadata_preserved(self) -> None:
        """Metadata is preserved in result."""
        result = CliResult.ok(
            {"data": 1},
            metadata={"duration_ms": 123},
        )

        assert result.metadata.get("duration_ms") == 123
```

#### Golden File Testing

Create `tests/cli/_fixtures/golden/op_list_json.json`:

```json
{
  "success": true,
  "data": {
    "operations": [
      {
        "id": "analyze.functions",
        "summary": "Analyze function metrics",
        "tags": ["analytics", "functions"]
      }
    ],
    "count": 1
  }
}
```

Test using golden files:

```python
# In tests/cli/integration/test_op_commands.py

def test_op_list_json_output(
    populated_context: CliTestContext,
    golden_path: Path,
) -> None:
    """Test op list JSON output matches golden file."""
    result = op_list_handler_structured(populated_context.operations)

    expected = json.loads((golden_path / "op_list_json.json").read_text())
    actual = json.loads(result.to_json())

    # Compare structure, not exact values
    assert actual["success"] == expected["success"]
    assert "operations" in actual["data"]
    assert "count" in actual["data"]
```

---

## Phase 4.5: Dynamic Operation Discovery

### Value Proposition

Currently, users must:
- Read documentation to discover operations
- Guess at parameter names and formats
- Manually look up schema definitions

Dynamic discovery provides:
- **Self-documenting CLI** — Introspect operations at runtime
- **IDE-like experience** — Tab completion knows all operations
- **Schema access** — Get JSON Schema for any operation
- **Example generation** — See usage examples inline

### Implementation

#### File: `src/codeintel/cli/introspection.py`

```python
"""Introspection utilities for CLI operations.

Provides runtime discovery of operations, their schemas, and examples.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import get_operation_registry


@dataclass(frozen=True)
class OperationInfo:
    """Detailed information about an operation.

    Parameters
    ----------
    operation_id
        Unique identifier.
    category
        Operation category.
    description
        Human-readable description.
    parameters
        Parameter specifications.
    examples
        Usage examples.
    requires_progress
        Whether progress is shown.
    retryable
        Whether operation is retryable.
    """

    operation_id: str
    category: str
    description: str
    parameters: list[dict[str, Any]]
    examples: list[str]
    requires_progress: bool
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return asdict(self)


def get_operation_info(operation_id: str) -> OperationInfo | None:
    """Get detailed information about an operation.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    OperationInfo | None
        Operation information or None if not found.
    """
    registry = get_operation_registry()
    spec = registry.get(operation_id)

    if spec is None:
        return None

    parameters = _extract_parameters(spec)
    examples = _generate_examples(spec)

    return OperationInfo(
        operation_id=spec.operation_id,
        category=spec.category.value,
        description=spec.description,
        parameters=parameters,
        examples=examples,
        requires_progress=spec.requires_progress,
        retryable=spec.retryable,
    )


def get_operation_schema(operation_id: str) -> dict[str, Any] | None:
    """Get JSON Schema for an operation's parameters.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    dict[str, Any] | None
        JSON Schema or None if not found.
    """
    registry = get_operation_registry()
    spec = registry.get(operation_id)

    if spec is None or spec.param_schema is None:
        return None

    return _schema_to_json_schema(spec.param_schema)


def list_operations_by_category() -> dict[str, list[str]]:
    """List operations grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Operations grouped by category name.
    """
    registry = get_operation_registry()
    result: dict[str, list[str]] = {}

    for spec in registry.list_operations():
        category = spec.category.value
        if category not in result:
            result[category] = []
        result[category].append(spec.operation_id)

    return result


def search_operations(query: str) -> list[OperationInfo]:
    """Search operations by ID or description.

    Parameters
    ----------
    query
        Search query.

    Returns
    -------
    list[OperationInfo]
        Matching operations.
    """
    registry = get_operation_registry()
    query_lower = query.lower()
    results = []

    for spec in registry.list_operations():
        if query_lower in spec.operation_id.lower():
            info = get_operation_info(spec.operation_id)
            if info:
                results.append(info)
        elif query_lower in spec.description.lower():
            info = get_operation_info(spec.operation_id)
            if info:
                results.append(info)

    return results


def _extract_parameters(spec: OperationSpec[Any]) -> list[dict[str, Any]]:
    """Extract parameter information from spec.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    list[dict[str, Any]]
        Parameter information.
    """
    if spec.param_schema is None:
        return []

    params = []
    for name, validator in spec.param_schema.validators.items():
        param_info: dict[str, Any] = {
            "name": name,
            "type": type(validator).__name__.replace("Validator", "").lower(),
            "required": True,  # Default to required
        }
        params.append(param_info)

    return params


def _generate_examples(spec: OperationSpec[Any]) -> list[str]:
    """Generate usage examples for an operation.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    list[str]
        Example command lines.
    """
    base = f"codeintel op call {spec.operation_id}"
    examples = [base]

    if spec.param_schema:
        param_str = " ".join(
            f"--{name}=VALUE"
            for name in spec.param_schema.validators.keys()
        )
        examples.append(f"{base} {param_str}")

    return examples


def _schema_to_json_schema(schema: Any) -> dict[str, Any]:
    """Convert validation schema to JSON Schema.

    Parameters
    ----------
    schema
        Validation schema.

    Returns
    -------
    dict[str, Any]
        JSON Schema.
    """
    # Basic conversion - can be extended for full schema support
    properties: dict[str, Any] = {}
    required = []

    if hasattr(schema, "validators"):
        for name, validator in schema.validators.items():
            prop: dict[str, Any] = {"type": "string"}

            validator_name = type(validator).__name__
            if "Int" in validator_name:
                prop["type"] = "integer"
            elif "Path" in validator_name:
                prop["type"] = "string"
                prop["format"] = "path"

            properties[name] = prop
            required.append(name)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
    }


__all__ = [
    "OperationInfo",
    "get_operation_info",
    "get_operation_schema",
    "list_operations_by_category",
    "search_operations",
]
```

#### Dynamic Shell Completion

Update `cli_completions.py` to use registry:

```python
def generate_dynamic_bash_completion() -> str:
    """Generate Bash completion with dynamic operation list.

    Returns
    -------
    str
        Bash completion script.
    """
    from codeintel.cli.operation_registry import get_operation_registry

    registry = get_operation_registry()
    operations = [spec.operation_id for spec in registry.list_operations()]

    return BASH_COMPLETION_TEMPLATE.replace(
        "# DYNAMIC_OPERATIONS",
        f'local operations="{" ".join(operations)}"',
    )
```

#### CLI Commands for Introspection

```python
@op_app.command()
def schema(
    operation_id: Annotated[str, Parameter(help="Operation ID")],
) -> None:
    """Show JSON Schema for an operation's parameters.

    Examples
    --------
    codeintel op schema analyze.functions
    """
    from codeintel.cli.introspection import get_operation_schema

    schema = get_operation_schema(operation_id)
    if schema is None:
        console.print(f"Operation not found: {operation_id}", style="error")
        raise SystemExit(1)

    print(json.dumps(schema, indent=2))


@op_app.command()
def info(
    operation_id: Annotated[str, Parameter(help="Operation ID")],
) -> None:
    """Show detailed information about an operation.

    Examples
    --------
    codeintel op info analyze.functions
    """
    from codeintel.cli.introspection import get_operation_info

    info = get_operation_info(operation_id)
    if info is None:
        console.print(f"Operation not found: {operation_id}", style="error")
        raise SystemExit(1)

    console.print(f"[heading]Operation: {info.operation_id}[/heading]")
    console.print(f"Category: {info.category}")
    console.print(f"Description: {info.description}")
    console.print()

    if info.parameters:
        console.print("[heading]Parameters:[/heading]")
        for param in info.parameters:
            console.print(f"  • {param['name']} ({param['type']})")

    if info.examples:
        console.print()
        console.print("[heading]Examples:[/heading]")
        for example in info.examples:
            console.print(f"  $ {example}")


@op_app.command()
def search(
    query: Annotated[str, Parameter(help="Search query")],
) -> None:
    """Search for operations by name or description.

    Examples
    --------
    codeintel op search coverage
    codeintel op search "build index"
    """
    from codeintel.cli.introspection import search_operations

    results = search_operations(query)
    if not results:
        console.print(f"No operations matching: {query}")
        return

    for op_info in results:
        console.print(f"[cyan]{op_info.operation_id}[/cyan]")
        console.print(f"  {op_info.description}")
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority | Effort |
|-------|----------|--------------|----------|--------|
| 4.1 Full Integration | 4-5 days | None | Critical | High |
| 4.2 Error Semantics | 2-3 days | 4.1 | High | Medium |
| 4.3 OpenTelemetry | 3-4 days | 4.1 | High | Medium |
| 4.4 Test Suite | 5-6 days | 4.1, 4.2 | High | High |
| 4.5 Discovery | 2-3 days | 4.1 | Medium | Low |

**Total estimated time: 16-21 days**

### Recommended Order

1. **Phase 4.1** (Full Integration) — Foundation for everything else
2. **Phase 4.2** (Error Semantics) — Essential for user experience
3. **Phase 4.3** (OpenTelemetry) — Can proceed in parallel with 4.4
4. **Phase 4.4** (Test Suite) — Validates all previous work
5. **Phase 4.5** (Discovery) — Polish and developer experience

### Parallel Workstreams

```
Week 1-2:     [====== Phase 4.1 ======]
Week 2-3:                [=== 4.2 ===]
Week 3-4:                [===== Phase 4.3 =====]
Week 2-4:                [======== Phase 4.4 ========]
Week 4-5:                              [=== 4.5 ===]
```

---

## Success Metrics

### Technical Quality

- [ ] All handlers use `OperationExecutor` (100% adoption)
- [ ] All errors use standardized error codes
- [ ] All operations have trace spans
- [ ] Test coverage ≥ 80% for CLI modules
- [ ] Zero `monkeypatch` usage in CLI tests

### Operational Excellence

- [ ] p95 operation latency tracked and < 500ms for read operations
- [ ] Error rates visible in metrics
- [ ] Correlation IDs in all error responses
- [ ] Trace context propagated to logs

### Developer Experience

- [ ] `codeintel op info <id>` works for all operations
- [ ] `codeintel op schema <id>` returns valid JSON Schema
- [ ] Shell completions include all registered operations
- [ ] Debug mode shows stack traces and internal state

### User Experience

- [ ] Consistent error message format across all commands
- [ ] Progress bars for operations > 2 seconds
- [ ] Clear fix suggestions in validation errors
- [ ] Retry behavior visible to users

---

## Migration Guide

### For Handler Authors

**Before Phase 4:**
```python
@build_app.command()
def status(output_format: OutputFormat = OutputFormat.TEXT) -> None:
    """Show build status."""
    runtime = resolve_runtime_options()
    result = build_status_handler_structured(runtime)
    if not result.success:
        console.print(result.error.detail, style="error")
        raise SystemExit(1)
    if output_format == OutputFormat.JSON:
        print(result.to_json())
    else:
        render_build_status(result.data)
```

**After Phase 4:**
```python
from codeintel.cli.executor import OperationCategory, OperationSpec, get_executor
from codeintel.cli.operation_registry import register_operation

BUILD_STATUS_SPEC = register_operation(
    OperationSpec(
        operation_id="build.status",
        handler=build_status_handler_structured,
        category=OperationCategory.READ,
        description="Show build target status",
    )
)

@build_app.command()
def status(output_format: OutputFormat = OutputFormat.TEXT) -> None:
    """Show build status."""
    runtime = resolve_runtime_options()
    executor = get_executor()
    result = executor.execute(
        BUILD_STATUS_SPEC,
        {"runtime": runtime},
        output_format=output_format,
    )
    if not result.result.success:
        raise SystemExit(1)
```

### For Test Authors

**Before Phase 4:**
```python
def test_op_list(monkeypatch):  # FORBIDDEN
    """Test op list command."""
    monkeypatch.setattr(...)
    result = runner.invoke(...)
```

**After Phase 4:**
```python
def test_op_list(populated_context: CliTestContext) -> None:
    """Test op list returns structured result."""
    result = op_list_handler_structured(populated_context.operations)

    assert result.success
    assert result.data.count == 3
```

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/cli/executor.py` | Unified operation execution pipeline |
| `src/codeintel/cli/operation_registry.py` | Central operation registration |
| `src/codeintel/cli/errors.py` | Standardized error types and codes |
| `src/codeintel/cli/telemetry.py` | OpenTelemetry integration |
| `src/codeintel/cli/introspection.py` | Runtime operation discovery |
| `tests/cli/conftest.py` | Shared test fixtures |
| `tests/cli/unit/test_executor.py` | Executor unit tests |
| `tests/cli/contract/test_cli_result_contract.py` | CliResult contract tests |
| `tests/cli/_fixtures/golden/*.json` | Golden files for output testing |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/cyclopts_app.py` | Add debug flag, introspection commands |
| `src/codeintel/cli/cyclopts_ops.py` | Use OperationExecutor |
| `src/codeintel/cli/cyclopts_build.py` | Use OperationExecutor |
| `src/codeintel/cli/cli_middleware.py` | Add TracingMiddleware integration |
| `src/codeintel/cli/cli_completions.py` | Dynamic operation completion |
| `tests/cli/_doubles/__init__.py` | Additional test doubles as needed |

---

*End of Phase 4 Implementation Plan*

