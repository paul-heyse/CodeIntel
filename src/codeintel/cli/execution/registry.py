"""Unified operation registry for CLI operations.

This module provides the single, canonical registry for all CLI operations.
Operations are registered here and discovered by:

- The @cli_command decorator
- The help system
- Programmatic execution via execute_operation()

Examples
--------
>>> from codeintel.cli.execution.registry import register_operation, OperationSpec
>>> from codeintel.cli.handlers.context import HandlerContext
>>> from codeintel.cli.core import CliResult
>>>
>>> def my_handler(ctx: HandlerContext) -> CliResult:  # doctest: +SKIP
...     return CliResult.ok({"status": "done"})
>>>
>>> spec = register_operation(
...     OperationSpec(  # doctest: +SKIP
...         operation_id="my.operation",
...         name="My Operation",
...         description="Does something useful",
...         handler=my_handler,
...         group="my",
...     )
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.core import CliResult
    from codeintel.cli.handlers.context import HandlerContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperationSpec:
    """Unified specification for a CLI operation.

    This is the single, canonical OperationSpec for all CLI operations.
    Core fields are required. Resource requirements default to True for
    backward compatibility. Execution hints are optional for future
    middleware integration.

    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list", "build.run").
    name
        Human-readable display name.
    description
        Help text describing the operation.
    handler
        Handler function to execute.
    group
        Command group (e.g., "jobs", "build").
    require_runtime
        Whether handler needs ResolvedRuntime.
    require_gateway
        Whether handler needs StorageGateway.
    require_graph_runtime
        Whether handler needs GraphRuntime.
    tags
        Optional tags for filtering/categorization.
    hidden
        If True, operation is hidden from help output.
    timeout
        Optional maximum execution time in seconds.
    retryable
        Whether the operation can be retried on failure.
    estimated_duration
        Optional estimated duration in seconds (for progress display).

    Examples
    --------
    >>> from codeintel.cli.execution.registry import OperationSpec
    >>> from codeintel.cli.handlers.context import HandlerContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def example_handler(ctx: HandlerContext) -> CliResult:  # doctest: +SKIP
    ...     return CliResult.ok({})
    >>>
    >>> spec = OperationSpec(  # doctest: +SKIP
    ...     operation_id="jobs.list",
    ...     name="List Jobs",
    ...     description="List background jobs",
    ...     handler=example_handler,
    ...     group="jobs",
    ...     require_runtime=False,
    ... )
    """

    # Core identification (required)
    operation_id: str
    name: str
    description: str
    handler: Callable[[HandlerContext], CliResult[Any]]
    group: str

    # Resource requirements
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False

    # Metadata
    tags: tuple[str, ...] = ()
    hidden: bool = False

    # Execution hints (optional, for future middleware integration)
    timeout: float | None = None
    retryable: bool = False
    estimated_duration: float | None = None


@dataclass
class OperationRegistry:
    """Central registry for all CLI operations.

    The registry maintains a mapping of operation IDs to their specifications.
    Operations can be registered, retrieved, and listed.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import OperationRegistry, OperationSpec
    >>> from codeintel.cli.handlers.context import HandlerContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def dummy_handler(ctx: HandlerContext) -> CliResult:  # doctest: +SKIP
    ...     return CliResult.ok({})
    >>>
    >>> registry = OperationRegistry()
    >>> spec = OperationSpec(  # doctest: +SKIP
    ...     operation_id="test.op",
    ...     name="Test",
    ...     description="Test operation",
    ...     handler=dummy_handler,
    ...     group="test",
    ... )
    >>> registry.register(spec)  # doctest: +SKIP
    >>> registry.get("test.op")  # doctest: +SKIP
    """

    _operations: dict[str, OperationSpec] = field(default_factory=dict)

    def register(self, spec: OperationSpec) -> OperationSpec:
        """Register an operation specification.

        Parameters
        ----------
        spec
            Operation specification to register.

        Returns
        -------
        OperationSpec
            The registered specification (for chaining).

        Raises
        ------
        ValueError
            If operation ID is already registered.
        """
        if spec.operation_id in self._operations:
            msg = f"Operation already registered: {spec.operation_id}"
            raise ValueError(msg)

        self._operations[spec.operation_id] = spec
        LOG.debug("Registered operation: %s", spec.operation_id)
        return spec

    def get(self, operation_id: str) -> OperationSpec | None:
        """Get an operation specification by ID.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec | None
            Specification if found, None otherwise.
        """
        return self._operations.get(operation_id)

    def require(self, operation_id: str) -> OperationSpec:
        """Get an operation specification, raising if not found.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec
            The operation specification.

        Raises
        ------
        KeyError
            If operation not found.
        """
        spec = self._operations.get(operation_id)
        if spec is None:
            msg = f"Operation not found: {operation_id}"
            raise KeyError(msg)
        return spec

    def list_operations(
        self,
        *,
        group: str | None = None,
        include_hidden: bool = False,
    ) -> list[OperationSpec]:
        """List registered operations.

        Parameters
        ----------
        group
            Optional group filter.
        include_hidden
            If True, include hidden operations.

        Returns
        -------
        list[OperationSpec]
            Matching operations sorted by operation_id.
        """
        ops = list(self._operations.values())

        if group is not None:
            ops = [op for op in ops if op.group == group]

        if not include_hidden:
            ops = [op for op in ops if not op.hidden]

        return sorted(ops, key=lambda op: op.operation_id)

    def list_groups(self) -> list[str]:
        """List all operation groups.

        Returns
        -------
        list[str]
            Sorted list of unique group names.
        """
        groups = {op.group for op in self._operations.values()}
        return sorted(groups)

    def unregister(self, operation_id: str) -> bool:
        """Unregister an operation.

        Parameters
        ----------
        operation_id
            Operation to remove.

        Returns
        -------
        bool
            True if operation was removed.
        """
        if operation_id in self._operations:
            del self._operations[operation_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()

    def __len__(self) -> int:
        """Return number of registered operations.

        Returns
        -------
        int
            Number of registered operations.
        """
        return len(self._operations)

    def __contains__(self, operation_id: str) -> bool:
        """Check if operation is registered.

        Parameters
        ----------
        operation_id
            Operation identifier to check.

        Returns
        -------
        bool
            True if operation is registered.
        """
        return operation_id in self._operations


# -----------------------------------------------------------------------------
# Global Registry
# -----------------------------------------------------------------------------

_REGISTRY: OperationRegistry | None = None


def get_registry() -> OperationRegistry:
    """Get the global operation registry.

    Create the registry on first access (lazy initialization).

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    global _REGISTRY  # noqa: PLW0603

    if _REGISTRY is None:
        _REGISTRY = OperationRegistry()

    return _REGISTRY


def register_operation(spec: OperationSpec) -> OperationSpec:
    """Register an operation with the global registry.

    Convenience function that gets the global registry and registers
    the operation.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    OperationSpec
        The registered specification.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import register_operation, OperationSpec
    >>> from codeintel.cli.handlers.context import HandlerContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def my_handler(ctx: HandlerContext) -> CliResult:  # doctest: +SKIP
    ...     return CliResult.ok({})
    >>>
    >>> register_operation(
    ...     OperationSpec(  # doctest: +SKIP
    ...         operation_id="my.op",
    ...         name="My Operation",
    ...         description="Does something",
    ...         handler=my_handler,
    ...         group="my",
    ...     )
    ... )
    """
    return get_registry().register(spec)


def reset_registry() -> None:
    """Reset the global registry (for testing only).

    WARNING: This function is for testing purposes only.
    Do not call in production code.
    """
    global _REGISTRY  # noqa: PLW0603
    _REGISTRY = None


def execute_operation(
    spec: OperationSpec,
    params: dict[str, Any],
) -> CliResult[Any]:
    """Execute an operation directly via its handler.

    This is the simple execution path that creates a HandlerContext and
    calls the handler. For programmatic execution of registered operations.

    Parameters
    ----------
    spec
        Operation specification (from registry).
    params
        Operation parameters dict.

    Returns
    -------
    CliResult[Any]
        Result from the handler.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import get_registry, execute_operation
    >>> spec = get_registry().get("some.operation")  # doctest: +SKIP
    >>> result = execute_operation(spec, {"param": "value"})  # doctest: +SKIP
    """
    # Import here to avoid circular imports
    from codeintel.cli.config import load_config  # noqa: PLC0415
    from codeintel.cli.handlers.context import HandlerContext  # noqa: PLC0415
    from codeintel.cli.rendering.types import OutputFormat  # noqa: PLC0415

    # Load config for context creation
    config = load_config(validate=False)

    # Create handler context
    ctx = HandlerContext(
        config=config,
        operation_id=spec.operation_id,
        output_format=OutputFormat.JSON,  # Default to JSON for programmatic use
        verbosity=0,
        _params=params,
    )

    try:
        # Execute handler
        return spec.handler(ctx)
    finally:
        ctx.close()


__all__ = [
    "OperationRegistry",
    "OperationSpec",
    "execute_operation",
    "get_registry",
    "register_operation",
    "reset_registry",
]
