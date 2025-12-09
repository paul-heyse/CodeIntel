"""Cyclopts-to-Executor adapter.

This module provides the bridge between Cyclopts command classes
and the unified OperationExecutor, ensuring all commands benefit from:
- Middleware (logging, metrics, tracing)
- Resilience (retries, circuit breakers)
- Progress tracking
- Plugin support
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from codeintel.cli.cli_render import get_renderer, render_cli_result
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.config import load_config
from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.execution.executor import (
    OperationCategory,
    OperationSpec,
    get_executor,
)
from codeintel.cli.handlers.base import setup_logging
from codeintel.cli.operation_registry import get_operation_registry
from codeintel.cli.results import CliResult

LOG = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Operation Decorator
# =============================================================================


def operation(
    operation_id: str,
    *,
    category: OperationCategory = OperationCategory.READ,
    description: str = "",
    retryable: bool = False,
) -> Callable[[Callable[P, CliResult[T]]], Callable[P, CliResult[T]]]:
    """Register a handler as an operation and route through the executor.

    This decorator:
    1. Creates an OperationSpec for the handler
    2. Registers it with the OperationRegistry
    3. Wraps the handler to route through the OperationExecutor

    Parameters
    ----------
    operation_id
        Unique operation identifier (e.g., "build.run").
    category
        Operation category for grouping.
    description
        Human-readable description.
    retryable
        Whether the operation can be retried.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @operation("build.run", category=OperationCategory.BUILD)
    ... def build_run_handler(ctx: HandlerContext) -> CliResult[BuildResult]:
    ...     ...
    """

    def decorator(handler: Callable[P, CliResult[T]]) -> Callable[P, CliResult[T]]:
        """Wrap handler with executor routing.

        Returns
        -------
        Callable[P, CliResult[T]]
            Wrapped handler function.
        """
        spec = OperationSpec(
            operation_id=operation_id,
            handler=handler,
            category=category,
            description=description or (handler.__doc__ or "").split("\n")[0],
            retryable=retryable,
        )

        # Register with the operation registry
        registry = get_operation_registry()
        registry.register(spec)

        @wraps(handler)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> CliResult[T]:
            """Route call through executor.

            Returns
            -------
            CliResult[T]
                Execution result.
            """
            executor = get_executor()
            params = _merge_args_to_params(handler, args, kwargs)
            return executor.execute(spec, params)  # type: ignore[return-value]

        # Attach spec and original handler for testing/introspection
        # Using non-private names to avoid SLF001 lint errors
        wrapper.operation_spec = spec  # type: ignore[attr-defined]
        wrapper.original_handler = handler  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _merge_args_to_params(
    handler: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge positional and keyword arguments into a params dict.

    Parameters
    ----------
    handler
        The handler function.
    args
        Positional arguments.
    kwargs
        Keyword arguments.

    Returns
    -------
    dict[str, Any]
        Merged parameters dictionary.
    """
    sig = inspect.signature(handler)
    params = dict(kwargs)

    for i, (name, _param) in enumerate(sig.parameters.items()):
        if i < len(args):
            params[name] = args[i]

    return params


# =============================================================================
# Cyclopts Adapter
# =============================================================================


class CycloptsAdapter:
    """Bridge Cyclopts command dataclasses to the operation executor.

    This adapter:
    1. Extracts parameters from Cyclopts command dataclass instances
    2. Sets up logging based on verbosity
    3. Builds handler context
    4. Routes through the executor (with middleware, resilience, etc.)
    5. Renders the result

    Parameters
    ----------
    operation_id
        Operation identifier.
    handler
        Handler function to invoke.

    Examples
    --------
    >>> @build_app.command(name="run")
    ... @dataclass
    ... class BuildRunCli:
    ...     targets: list[str] | None = None
    ...     verbose: int = 0
    ...
    ...     def __call__(self) -> None:
    ...         CycloptsAdapter("build.run", build_run_handler)(self)
    """

    def __init__(
        self,
        operation_id: str,
        handler: Callable[..., CliResult[Any]],
        *,
        category: OperationCategory = OperationCategory.READ,
    ) -> None:
        """Initialize the adapter."""
        self._operation_id = operation_id
        self._handler = handler
        self._category = category
        # Check if handler has a pre-registered spec
        self._spec: OperationSpec[Any] | None = getattr(handler, "operation_spec", None)

    def __call__(self, command: object) -> None:
        """Execute the command through the adapter.

        Parameters
        ----------
        command
            Cyclopts command dataclass instance.
        """
        # Extract parameters from command dataclass
        params = self._extract_params(command)
        verbosity = int(params.pop("verbose", params.pop("verbosity", 0)))
        output_format_str = str(params.get("output_format", "text"))
        output_format = OutputFormat.JSON if output_format_str == "json" else OutputFormat.TEXT

        # Load config and setup logging
        config = load_config(validate=False)
        setup_logging(verbosity, config=config)

        LOG.debug(
            "CycloptsAdapter executing operation=%s params=%s",
            self._operation_id,
            params,
        )

        # Execute through executor or directly
        result: CliResult[Any]
        if self._spec is not None:
            executor = get_executor()
            exec_result = executor.execute(self._spec, params)
            # ExecutionResult wraps CliResult in .result attribute
            result = exec_result.result
        else:
            # Build context and call handler directly
            ctx = ExecutionContext.for_sync(self._operation_id, params)
            result = self._handler(ctx, **params)

        # Render the result
        renderer = get_renderer(output_format)
        render_cli_result(result, renderer)

    @staticmethod
    def _extract_params(command: object) -> dict[str, Any]:
        """Extract parameters from a Cyclopts command dataclass.

        Parameters
        ----------
        command
            Command dataclass instance.

        Returns
        -------
        dict[str, Any]
            Extracted parameters.
        """
        if not is_dataclass(command):
            return {}

        params: dict[str, Any] = {}
        for field_obj in fields(command):
            name = field_obj.name
            value = getattr(command, name, None)
            # Skip None values to use handler defaults
            if value is not None:
                params[name] = value

        return params


# =============================================================================
# Helper Functions
# =============================================================================


def adapt_cyclopts_command(
    operation_id: str,
    handler: Callable[..., CliResult[Any]],
    command: object,
    *,
    category: OperationCategory = OperationCategory.READ,
) -> None:
    """Adapt and execute a Cyclopts command through the executor.

    Parameters
    ----------
    operation_id
        Operation identifier.
    handler
        Handler function.
    command
        Cyclopts command dataclass instance.
    category
        Operation category.
    """
    adapter = CycloptsAdapter(operation_id, handler, category=category)
    adapter(command)


__all__ = [
    "CycloptsAdapter",
    "adapt_cyclopts_command",
    "operation",
]
