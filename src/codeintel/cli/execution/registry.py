"""Unified operation registry for CLI operations.

This module provides the single, canonical registry for all CLI operations.
Operations are registered here and discovered by:

- The @cli_command decorator
- The help system
- Programmatic execution via execute_operation()

Examples
--------
>>> from codeintel.cli.execution.registry import register_operation, OperationSpec
>>> from codeintel.cli.context import CommandContext
>>> from codeintel.cli.core import CliResult
>>>
>>> def my_handler(ctx: CommandContext) -> CliResult:  # doctest: +SKIP
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

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from codeintel.cli.rendering.types import OutputFormat
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from codeintel.cli.context import CommandContext, CommandContextBuilder
    from codeintel.cli.core import CliResult

LOG = logging.getLogger(__name__)


class _ContextModule(Protocol):
    """Protocol describing the CLI context module exports."""

    CommandContextBuilder: type[CommandContextBuilder]


def _load_context_module() -> _ContextModule:
    """Import the CLI context module lazily to avoid import cycles."""
    module = importlib.import_module("codeintel.cli.context")
    return cast("_ContextModule", module)


@dataclass(frozen=True)
class OperationSpec:
    """Unified specification for a CLI operation.

    This is the single, canonical OperationSpec for all CLI operations.
    Core fields are required. Resource requirements default to False;
    operations must explicitly declare their requirements. Execution hints
    and serving integration fields are optional.

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
    serving_op_id
        ID in the serving operations catalog (for bridged operations).
    http_path
        HTTP endpoint path (for serving operations).
    tool_name
        MCP tool name (for serving operations).
    backend_method
        Backend method name (for serving operations).

    Examples
    --------
    >>> from codeintel.cli.execution.registry import OperationSpec
    >>> from codeintel.cli.context import CommandContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def example_handler(ctx: CommandContext) -> CliResult:  # doctest: +SKIP
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
    handler: Callable[[CommandContext], CliResult[Any]]
    group: str

    # Resource requirements (explicitly declare what each operation needs)
    require_runtime: bool = False
    require_gateway: bool = False
    require_graph_runtime: bool = False

    # Metadata
    tags: tuple[str, ...] = ()
    hidden: bool = False

    # Execution hints (optional, for future middleware integration)
    timeout: float | None = None
    retryable: bool = False
    estimated_duration: float | None = None

    # Serving integration (optional, for bridged operations)
    serving_op_id: str | None = None
    http_path: str | None = None
    tool_name: str | None = None
    backend_method: str | None = None

    def to_dict(self, *, include_handler: bool = False) -> dict[str, object]:
        """Convert to dictionary representation.

        Create a serializable dictionary of the operation specification.
        By default, excludes the handler since it's not serializable.

        Parameters
        ----------
        include_handler
            If True, include handler reference in output. Default False.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the specification.

        Examples
        --------
        >>> from codeintel.cli.execution.registry import OperationSpec
        >>> spec = OperationSpec(  # doctest: +SKIP
        ...     operation_id="test.op",
        ...     name="Test",
        ...     description="Test operation",
        ...     handler=lambda ctx: None,
        ...     group="test",
        ... )
        >>> d = spec.to_dict()  # doctest: +SKIP
        >>> "handler" in d  # doctest: +SKIP
        False
        """
        result: dict[str, object] = {
            "operation_id": self.operation_id,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "require_runtime": self.require_runtime,
            "require_gateway": self.require_gateway,
            "require_graph_runtime": self.require_graph_runtime,
            "tags": list(self.tags),
            "hidden": self.hidden,
        }

        # Add optional execution hints if set
        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.retryable:
            result["retryable"] = self.retryable
        if self.estimated_duration is not None:
            result["estimated_duration"] = self.estimated_duration

        # Add serving integration fields if set
        if self.serving_op_id is not None:
            result["serving_op_id"] = self.serving_op_id
        if self.http_path is not None:
            result["http_path"] = self.http_path
        if self.tool_name is not None:
            result["tool_name"] = self.tool_name
        if self.backend_method is not None:
            result["backend_method"] = self.backend_method

        # Include handler reference if requested
        if include_handler:
            handler_name = getattr(self.handler, "__name__", repr(self.handler))
            result["handler"] = handler_name

        return result


@dataclass
class OperationRegistry:
    """Central registry for all CLI operations.

    The registry maintains a mapping of operation IDs to their specifications.
    Operations can be registered, retrieved, and listed.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import OperationRegistry, OperationSpec
    >>> from codeintel.cli.context import CommandContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def dummy_handler(ctx: CommandContext) -> CliResult:  # doctest: +SKIP
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

    def __contains__(self, operation_id: object) -> bool:
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
        if not isinstance(operation_id, str):
            return False
        return operation_id in self._operations

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over registered operation IDs.

        Returns
        -------
        Iterator[str]
            Iterator over registered operation identifiers.
        """
        return iter(self._operations.keys())


# -----------------------------------------------------------------------------
# Global Registry
# -----------------------------------------------------------------------------


class OperationRegistryHolder(SingletonHolder[OperationRegistry]):
    """Singleton holder for the CLI OperationRegistry."""


def get_registry() -> OperationRegistry:
    """Get the global operation registry.

    Create the registry on first access (lazy initialization).

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    registry = OperationRegistryHolder.get(OperationRegistry)
    if not registry.list_operations():
        # Import command modules to trigger registration side effects.
        importlib.import_module("codeintel.cli.commands")

    return registry


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
    >>> from codeintel.cli.context import CommandContext
    >>> from codeintel.cli.core import CliResult
    >>>
    >>> def my_handler(ctx: CommandContext) -> CliResult:  # doctest: +SKIP
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
    OperationRegistryHolder.reset()


def create_spec_from_serving_operation(
    serving_op_id: str,
    handler: Callable[[CommandContext], CliResult[Any]],
    *,
    cli_operation_id: str | None = None,
    group: str | None = None,
) -> OperationSpec:
    """Create an OperationSpec from a serving operation.

    Bridge function that looks up a serving operation by ID and creates
    an OperationSpec with the relevant metadata populated.

    Parameters
    ----------
    serving_op_id
        ID in the serving operations catalog (e.g., "function.summary").
    handler
        Handler function to execute for this CLI operation.
    cli_operation_id
        CLI operation ID. Defaults to "op.<serving_op_id>" if not provided.
    group
        CLI group. Defaults to "op" if not provided.

    Returns
    -------
    OperationSpec
        Specification with serving metadata populated.

    Raises
    ------
    ValueError
        If the serving operation is not found.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import create_spec_from_serving_operation
    >>> spec = create_spec_from_serving_operation(  # doctest: +SKIP
    ...     "function.summary",
    ...     my_handler,
    ... )
    >>> spec.serving_op_id
    'function.summary'
    """
    serving_module = importlib.import_module("codeintel.serving.operations.catalog")
    serving_op = serving_module.get_operation(serving_op_id)
    if serving_op is None:
        msg = f"Serving operation not found: {serving_op_id}"
        raise ValueError(msg)

    effective_cli_id = cli_operation_id or f"op.{serving_op_id.replace('.', '-')}"
    effective_group = group or "op"

    return OperationSpec(
        operation_id=effective_cli_id,
        name=serving_op.summary,
        description=serving_op.description or serving_op.summary,
        handler=handler,
        group=effective_group,
        require_runtime=True,
        require_gateway=True,
        require_graph_runtime=bool(serving_op.required_graphs),
        serving_op_id=serving_op_id,
        http_path=serving_op.http_path,
        tool_name=serving_op.tool_name,
        backend_method=serving_op.backend_method,
    )


def execute_operation(
    spec: OperationSpec,
    params: dict[str, Any],
) -> CliResult[Any]:
    """Execute an operation directly via its handler.

    This is the simple execution path that creates a CommandContext and
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
    builder = (
        _load_context_module().CommandContextBuilder()
        .with_params(params)
        .with_output_format(OutputFormat.JSON)  # Default to JSON for programmatic use
        .with_verbosity(0)
        .with_operation_id(spec.operation_id)
    )

    # Apply resource requirements from spec
    if spec.require_runtime:
        builder = builder.with_runtime()

    if spec.require_gateway:
        builder = builder.with_storage()

    with builder.build() as ctx:
        return spec.handler(ctx)


__all__ = [
    "OperationRegistry",
    "OperationSpec",
    "create_spec_from_serving_operation",
    "execute_operation",
    "get_registry",
    "register_operation",
    "reset_registry",
]
