"""Handler protocol and enhanced context.

This module defines:

- HandlerProtocol: Contract for all CLI handlers
- EnhancedHandlerContext: Context with lazy gateway and runtime access
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.cli.results import CliResult

T = TypeVar("T")


@dataclass
class EnhancedHandlerContext:
    """Enhanced context for CLI handlers with lazy resource access.

    Provide lazy access to gateway and graph_runtime to avoid opening
    connections unnecessarily. Resources are opened on first access
    and closed via the close() method.

    Parameters
    ----------
    config
        CLI configuration.
    runtime
        Resolved project runtime.
    params
        Operation-specific parameters.
    verbosity
        Verbosity level (0=warnings, 1=info, 2+=debug).

    Examples
    --------
    >>> # In a handler:
    >>> def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
    ...     ctx.logger.info("Starting")
    ...     data = ctx.gateway.execute("SELECT * FROM table")  # doctest: +SKIP
    ...     return CliResult.ok(MyData(data))  # doctest: +SKIP
    """

    config: CliConfig
    runtime: ResolvedRuntime
    params: Mapping[str, object] = field(default_factory=dict)
    verbosity: int = 0

    # Private fields for lazy initialization
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _operation_name: str = field(default="handler", repr=False)

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this handler.

        Returns
        -------
        logging.Logger
            Logger with handler-specific name.
        """
        return logging.getLogger(f"codeintel.cli.handlers.{self._operation_name}")

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (lazy).

        Gateway is opened on first access. The context manages lifecycle.

        Returns
        -------
        StorageGateway
            Open storage gateway.
        """
        if self._gateway is None:
            storage_config = StorageConfig(db_path=self.runtime.db_path, read_only=True)
            self._gateway = open_gateway(storage_config)
        return self._gateway

    @property
    def graph_runtime(self) -> GraphRuntime:
        """Get graph runtime (lazy).

        Returns
        -------
        GraphRuntime
            Graph runtime for graph operations.
        """
        if self._graph_runtime is None:
            options = GraphRuntimeOptions(snapshot=self.runtime.snapshot)
            self._graph_runtime = build_graph_runtime(
                gateway=self.gateway,
                options=options,
            )
        return self._graph_runtime

    @property
    def db_path(self) -> Path:
        """Shortcut to database path.

        Returns
        -------
        Path
            Path to DuckDB database.
        """
        return self.runtime.db_path

    @property
    def output_format(self) -> str:
        """Get output format from config.

        Returns
        -------
        str
            Output format ('text' or 'json').
        """
        return self.config.output_format

    @property
    def color_enabled(self) -> bool:
        """Check if color output is enabled.

        Returns
        -------
        bool
            True if color is enabled.
        """
        return self.config.color

    def close(self) -> None:
        """Close managed resources.

        Called automatically when using as_context_manager() or
        should be called explicitly after handler execution.
        """
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None
        self._graph_runtime = None

    @contextmanager
    def as_context_manager(self) -> Iterator[EnhancedHandlerContext]:
        """Use context as a context manager for automatic cleanup.

        Yields
        ------
        EnhancedHandlerContext
            Self for use in with block.

        Examples
        --------
        >>> with ctx.as_context_manager():  # doctest: +SKIP
        ...     result = handler(ctx)  # doctest: +SKIP
        """
        try:
            yield self
        finally:
            self.close()


class HandlerProtocol(Protocol[T]):
    """Protocol for CLI handler functions.

    All handlers must:

    1. Accept EnhancedHandlerContext as their only argument
    2. Return CliResult[T] (never None, never raise for expected errors)
    3. Never write to stdout/stderr directly
    4. Never call sys.exit()

    Unexpected exceptions (bugs) may propagate; expected errors
    should return CliResult.fail() with appropriate ProblemDetail.

    Examples
    --------
    >>> def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:  # doctest: +SKIP
    ...     if not ctx.params.get("required"):  # doctest: +SKIP
    ...         return CliResult.fail(ProblemDetail(...))  # doctest: +SKIP
    ...     data = compute_result(ctx.gateway)  # doctest: +SKIP
    ...     return CliResult.ok(data)  # doctest: +SKIP
    """

    def __call__(self, ctx: EnhancedHandlerContext) -> CliResult[T]:
        """Execute the handler.

        Parameters
        ----------
        ctx
            Handler context with config, runtime, params.

        Returns
        -------
        CliResult[T]
            Success or failure result. Never None.
        """
        ...


@contextmanager
def handler_context(
    config: CliConfig,
    runtime: ResolvedRuntime,
    params: Mapping[str, object] | None = None,
    *,
    verbosity: int = 0,
    operation_name: str = "handler",
) -> Iterator[EnhancedHandlerContext]:
    """Create handler context with automatic resource cleanup.

    Parameters
    ----------
    config
        CLI configuration.
    runtime
        Resolved runtime.
    params
        Operation parameters.
    verbosity
        Verbosity level.
    operation_name
        Name for logging.

    Yields
    ------
    EnhancedHandlerContext
        Context for handler use.

    Examples
    --------
    >>> with handler_context(config, runtime, {"key": "value"}) as ctx:  # doctest: +SKIP
    ...     result = my_handler(ctx)  # doctest: +SKIP
    """
    ctx = EnhancedHandlerContext(
        config=config,
        runtime=runtime,
        params=params or {},
        verbosity=verbosity,
        _operation_name=operation_name,
    )
    try:
        yield ctx
    finally:
        ctx.close()


__all__ = [
    "EnhancedHandlerContext",
    "HandlerProtocol",
    "handler_context",
]
