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
>>> def my_handler(ctx: CommandContext) -> CliResult:
...     return CliResult.ok({"status": "done"})
>>>
>>> spec = register_operation(
...     OperationSpec(
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
import sys
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
    """Import the CLI context module lazily to avoid import cycles.

    Returns
    -------
    _ContextModule
        Imported context module providing CommandContextBuilder.
    """
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
    >>> def example_handler(ctx: CommandContext) -> CliResult:
    ...     return CliResult.ok({})
    >>>
    >>> spec = OperationSpec(
    ...     operation_id="jobs.list",
    ...     name="List Jobs",
    ...     description="List background jobs",
    ...     handler=example_handler,
    ...     group="jobs",
    ...     require_runtime=False,
    ... )
    """

    operation_id: str
    name: str
    description: str
    handler: Callable[[CommandContext], CliResult[Any]]
    group: str

    require_runtime: bool = False
    require_gateway: bool = False
    require_graph_runtime: bool = False

    tags: tuple[str, ...] = ()
    hidden: bool = False

    timeout: float | None = None
    retryable: bool = False
    estimated_duration: float | None = None

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
        >>> spec = OperationSpec(
        ...     operation_id="test.op",
        ...     name="Test",
        ...     description="Test operation",
        ...     handler=lambda ctx: None,
        ...     group="test",
        ... )
        >>> d = spec.to_dict()
        >>> "handler" in d
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

        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.retryable:
            result["retryable"] = self.retryable
        if self.estimated_duration is not None:
            result["estimated_duration"] = self.estimated_duration

        if self.serving_op_id is not None:
            result["serving_op_id"] = self.serving_op_id
        if self.http_path is not None:
            result["http_path"] = self.http_path
        if self.tool_name is not None:
            result["tool_name"] = self.tool_name
        if self.backend_method is not None:
            result["backend_method"] = self.backend_method

        if include_handler:
            handler_name = getattr(self.handler, "__name__", repr(self.handler))
            result["handler"] = handler_name

        return result


@dataclass(frozen=True)
class OperationAlias:
    """Alias mapping for legacy operation identifiers.

    Parameters
    ----------
    alias_id
        Legacy operation identifier.
    target_id
        Canonical operation identifier.
    deprecated
        Whether the alias is deprecated.
    note
        Optional deprecation guidance.
    """

    alias_id: str
    target_id: str
    deprecated: bool = True
    note: str | None = None


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
    >>> def dummy_handler(ctx: CommandContext) -> CliResult:
    ...     return CliResult.ok({})
    >>>
    >>> registry = OperationRegistry()
    >>> spec = OperationSpec(
    ...     operation_id="test.op",
    ...     name="Test",
    ...     description="Test operation",
    ...     handler=dummy_handler,
    ...     group="test",
    ... )
    >>> registry.register(spec)
    >>> registry.get("test.op")
    """

    _operations: dict[str, OperationSpec] = field(default_factory=dict)
    _aliases: dict[str, OperationAlias] = field(default_factory=dict)
    _aliases_by_target: dict[str, set[str]] = field(default_factory=dict)

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
        if spec.operation_id in self._aliases:
            msg = f"Operation ID conflicts with registered alias: {spec.operation_id}"
            raise ValueError(msg)

        self._operations[spec.operation_id] = spec
        LOG.debug("Registered operation: %s", spec.operation_id)
        return spec

    def register_alias(self, alias: OperationAlias) -> OperationAlias:
        """Register a legacy alias for a canonical operation.

        Parameters
        ----------
        alias
            Alias metadata to register.

        Returns
        -------
        OperationAlias
            Registered alias metadata.

        Raises
        ------
        ValueError
            If alias conflicts with an existing operation, alias, or missing target.
        """
        if alias.alias_id in self._operations:
            msg = f"Alias ID conflicts with operation: {alias.alias_id}"
            raise ValueError(msg)
        if alias.alias_id in self._aliases:
            msg = f"Alias already registered: {alias.alias_id}"
            raise ValueError(msg)
        if alias.target_id not in self._operations:
            msg = f"Alias target not registered: {alias.target_id}"
            raise ValueError(msg)

        self._aliases[alias.alias_id] = alias
        self._aliases_by_target.setdefault(alias.target_id, set()).add(alias.alias_id)
        LOG.debug("Registered alias: %s -> %s", alias.alias_id, alias.target_id)
        return alias

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
        spec = self._operations.get(operation_id)
        if spec is not None:
            return spec
        alias = self._aliases.get(operation_id)
        if alias is None:
            return None
        return self._operations.get(alias.target_id)

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
        spec = self.get(operation_id)
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

    def list_aliases(self, *, target_id: str | None = None) -> list[OperationAlias]:
        """List registered aliases.

        Parameters
        ----------
        target_id
            Optional canonical operation ID to filter aliases.

        Returns
        -------
        list[OperationAlias]
            Registered aliases, sorted by alias ID.
        """
        aliases = list(self._aliases.values())
        if target_id is not None:
            aliases = [alias for alias in aliases if alias.target_id == target_id]
        return sorted(aliases, key=lambda alias: alias.alias_id)

    def aliases_for(self, operation_id: str) -> tuple[OperationAlias, ...]:
        """Return aliases registered for a canonical operation.

        Parameters
        ----------
        operation_id
            Canonical operation identifier.

        Returns
        -------
        tuple[OperationAlias, ...]
            Alias metadata for the canonical operation.
        """
        aliases = self._aliases_by_target.get(operation_id, set())
        if not aliases:
            return ()
        return tuple(self._aliases[alias_id] for alias_id in sorted(aliases))

    def is_alias(self, operation_id: str) -> bool:
        """Return True when operation_id is a registered alias.

        Returns
        -------
        bool
            True if operation_id is a registered alias.
        """
        return operation_id in self._aliases

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
        if operation_id in self._aliases:
            alias = self._aliases.pop(operation_id)
            targets = self._aliases_by_target.get(alias.target_id)
            if targets is not None:
                targets.discard(operation_id)
                if not targets:
                    self._aliases_by_target.pop(alias.target_id, None)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()
        self._aliases.clear()
        self._aliases_by_target.clear()

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
        return operation_id in self._operations or operation_id in self._aliases

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over registered operation IDs.

        Returns
        -------
        Iterator[str]
            Iterator over registered operation identifiers.
        """
        return iter(self._operations.keys())


class OperationRegistryHolder(SingletonHolder[OperationRegistry]):
    """Singleton holder for the CLI OperationRegistry."""


@dataclass
class _BootstrapState:
    in_progress: bool = False


_BOOTSTRAP_STATE = _BootstrapState()


def _commands_module_is_initializing() -> bool:
    module = sys.modules.get("codeintel.cli.commands")
    if module is None:
        return False
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


def get_registry() -> OperationRegistry:
    """Get the global operation registry.

    Create the registry on first access (lazy initialization).

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    registry = OperationRegistryHolder.get(OperationRegistry)
    if registry.list_operations():
        _register_default_aliases(registry)
        return registry

    if _BOOTSTRAP_STATE.in_progress:
        return registry

    _BOOTSTRAP_STATE.in_progress = True
    try:
        importlib.import_module("codeintel.cli.commands")
        if registry.list_operations():
            _register_default_aliases(registry)
            return registry
        if _commands_module_is_initializing():
            return registry

        for module_name in list(sys.modules):
            is_commands = module_name == "codeintel.cli.commands"
            is_submodule = module_name.startswith("codeintel.cli.commands.")
            if is_commands or is_submodule:
                sys.modules.pop(module_name, None)

        importlib.import_module("codeintel.cli.commands")
    finally:
        _BOOTSTRAP_STATE.in_progress = False

    _register_default_aliases(registry)
    return registry


def _register_default_aliases(registry: OperationRegistry) -> None:
    for alias in _default_aliases():
        try:
            registry.register_alias(alias)
        except ValueError:
            LOG.debug("Skipping alias registration for %s", alias.alias_id)


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
    >>> def my_handler(ctx: CommandContext) -> CliResult:
    ...     return CliResult.ok({})
    >>>
    >>> register_operation(
    ...     OperationSpec(
    ...         operation_id="my.op",
    ...         name="My Operation",
    ...         description="Does something",
    ...         handler=my_handler,
    ...         group="my",
    ...     )
    ... )
    """
    return get_registry().register(spec)


def register_alias(
    alias_id: str,
    target_id: str,
    *,
    deprecated: bool = True,
    note: str | None = None,
) -> OperationAlias:
    """Register an alias for a canonical operation.

    Parameters
    ----------
    alias_id
        Legacy operation identifier.
    target_id
        Canonical operation identifier.
    deprecated
        Whether the alias is deprecated.
    note
        Optional deprecation guidance.

    Returns
    -------
    OperationAlias
        Registered alias metadata.
    """
    alias = OperationAlias(
        alias_id=alias_id,
        target_id=target_id,
        deprecated=deprecated,
        note=note,
    )
    return get_registry().register_alias(alias)


def reset_registry() -> None:
    """Reset the global registry (for testing only).

    WARNING: This function is for testing purposes only.
    Do not call in production code.
    """
    OperationRegistryHolder.reset()


def _default_aliases() -> tuple[OperationAlias, ...]:
    return (
        OperationAlias(
            alias_id="datasets.list",
            target_id="dataset.list",
            note="Use dataset.list.",
        ),
        OperationAlias(
            alias_id="datasets.describe",
            target_id="dataset.describe",
            note="Use dataset.describe.",
        ),
        OperationAlias(
            alias_id="datasets.verify",
            target_id="dataset.verify",
            note="Use dataset.verify.",
        ),
        OperationAlias(
            alias_id="datasets.info",
            target_id="dataset.info",
            note="Use dataset.info.",
        ),
        OperationAlias(
            alias_id="datasets.flow",
            target_id="dataset.flow",
            note="Use dataset.flow.",
        ),
        OperationAlias(
            alias_id="datasets.constraints",
            target_id="dataset.constraints",
            note="Use dataset.constraints.",
        ),
        OperationAlias(
            alias_id="graphs.targets.list",
            target_id="graph.targets.list",
            note="Use graph.targets.list.",
        ),
        OperationAlias(
            alias_id="graphs.targets.plan",
            target_id="graph.targets.plan",
            note="Use graph.targets.plan.",
        ),
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
    >>> spec = get_registry().get("some.operation")
    >>> result = execute_operation(spec, {"param": "value"})
    """
    builder = (
        _load_context_module()
        .CommandContextBuilder()
        .with_params(params)
        .with_output_format(OutputFormat.JSON)
        .with_verbosity(0)
        .with_operation_id(spec.operation_id)
    )

    if spec.require_runtime:
        builder = builder.with_runtime()

    if spec.require_gateway:
        builder = builder.with_storage()

    with builder.build() as ctx:
        return spec.handler(ctx)


__all__ = [
    "OperationAlias",
    "OperationRegistry",
    "OperationSpec",
    "execute_operation",
    "get_registry",
    "register_alias",
    "register_operation",
    "reset_registry",
]
