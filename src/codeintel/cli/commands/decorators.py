"""Declarative command binding via @cli_command decorator.

This module provides the @cli_command decorator that eliminates boilerplate
from CLI command classes. It supports two patterns:

**Legacy Pattern (handler-based)**:

- Handler function receives HandlerContext
- Use `config=CommandConfig(...)` for resource requirements
- Command class is a regular dataclass

**New Pattern (Command[T]-based)**:

- Command extends `Command[T]` from `cli.core.command`
- Implements `execute(self, deps: Deps) -> CliResult[T]`
- Uses `require_storage` and `require_serving` parameters
- Full end-to-end type safety

Both patterns are supported during migration. Prefer the new pattern for
new commands.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeGuard, TypeVar, cast

from codeintel.cli.commands._common import SharedFlags
from codeintel.cli.context import CommandContextBuilder
from codeintel.cli.context_compat import (
    deps_from_command_context,
    handler_context_from_command_context,
)
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.deps.compat import deps_from_handler_context
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.service import get_renderer
from codeintel.cli.rendering.types import OutputFormat

LOG = logging.getLogger(__name__)

# Runtime type for command dataclass instances (legacy pattern)
CommandInstance = object


class _DataclassCommand(Protocol):
    __dataclass_fields__: dict[str, dataclasses.Field[object]]

    def __init__(self, **kwargs: object) -> None: ...


def _is_dataclass_command_type(cls: type[object]) -> TypeGuard[type[_DataclassCommand]]:
    """Return True when cls is a dataclass command.

    Returns
    -------
    bool
        True when cls is a dataclass Command type.
    """
    return dataclasses.is_dataclass(cls)


CommandType = TypeVar("CommandType", bound=Command[Any])


@dataclass(frozen=True)
class CommandConfig:
    """Configuration for @cli_command decorator.

    Bundles resource requirements and metadata to reduce function argument count.

    Parameters
    ----------
    require_runtime
        Whether handler needs ResolvedRuntime.
    require_gateway
        Whether handler needs StorageGateway.
    require_graph_runtime
        Whether handler needs GraphRuntime.
    description
        Optional description (defaults to class docstring).
    """

    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    description: str | None = None


# Default configuration
DEFAULT_CONFIG = CommandConfig()

# Fields that are standard infrastructure, not command params
# Note: repo, repo_root, commit are NOT excluded because some commands
# (like history.timeseries) use them as actual parameters
_INFRASTRUCTURE_FIELDS = frozenset(
    {
        "output_format",
        "verbose",
        "json",
        "project",
        "project_root",  # Used for runtime resolution
        "db_path",
        "database_path",
        "index_path",
        "flags",  # SharedFlags mixin field
    }
)


def cli_command[T, R](
    operation_id: str,
    *,
    handler: Callable[[HandlerContext], CliResult[R]] | None = None,
    config: CommandConfig | None = None,
    require_storage: bool = False,
    require_serving: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Decorate CLI command dataclasses with automatic execution.

    Support two patterns:

    **Legacy Pattern** (handler= provided):
    Uses HandlerContext-based handlers with CommandConfig.

    **New Pattern** (Command[T] subclass, no handler):
    Command implements execute(deps) directly.

    Parameters
    ----------
    operation_id
        Unique operation identifier (e.g., "jobs.list").
    handler
        Legacy handler function to invoke. If None, command must extend Command[T].
    config
        Legacy command configuration (resource requirements, description).
        Defaults to requiring runtime and gateway. Ignored for new pattern.
    require_storage
        For new pattern: whether command needs storage access via deps.storage.
    require_serving
        For new pattern: whether command needs serving access via deps.serving.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Class decorator.

    Examples
    --------
    Legacy pattern with handler:

    >>> from codeintel.cli.handlers.context import HandlerContext
    >>> from codeintel.cli.core import CliResult
    >>> def my_handler(ctx: HandlerContext) -> CliResult[dict[str, str]]:
    ...     return CliResult.ok({"status": "done"})
    >>> # @cli_command("my.op", handler=my_handler)
    >>> # @dataclass
    >>> # class MyCommand:
    >>> #     name: str = "default"

    New pattern with Command[T]:

    >>> # @cli_command("jobs.list", require_storage=False)
    >>> # @dataclass(frozen=True)
    >>> # class ListJobs(Command[ListResult[JobInfo]]):
    >>> #     limit: int = 20
    >>> #
    >>> #     def execute(self, deps: Deps) -> CliResult[ListResult[JobInfo]]:
    >>> #         ...
    """

    def decorator(cls: type[T]) -> type[T]:
        # Detect pattern based on whether class is Command subclass
        is_new_pattern = _is_command_subclass(cls)

        if is_new_pattern and handler is None:
            command_cls = cast("type[Command[Any]]", cls)
            decorated = _decorate_new_style(
                command_cls,
                operation_id,
                require_storage=require_storage,
                require_serving=require_serving,
            )
            return cast("type[T]", decorated)

        if handler is not None:
            return _decorate_legacy(cls, operation_id, handler, config)

        msg = (
            f"@cli_command on {cls.__name__}: must provide handler= for legacy "
            "pattern or extend Command[T] for new pattern"
        )
        raise TypeError(msg)

    return decorator


def _is_command_subclass(cls: type[object]) -> bool:
    """Check if class is a Command[T] subclass.

    Parameters
    ----------
    cls
        Class to check.

    Returns
    -------
    bool
        True if cls extends Command[T].
    """
    try:
        # Check if it's a subclass of Command
        # Use __mro__ to avoid issues with generics
        return any(
            getattr(base, "__name__", "") == "Command"
            and getattr(base, "__module__", "").startswith("codeintel.cli")
            for base in cls.__mro__
            if base is not object
        )
    except TypeError:
        return False


def _decorate_legacy[T, R](
    cls: type[T],
    operation_id: str,
    handler: Callable[[HandlerContext], CliResult[R]],
    config: CommandConfig | None,
) -> type[T]:
    """Decorate with legacy handler pattern.

    Parameters
    ----------
    cls
        Command class.
    operation_id
        Operation identifier.
    handler
        Handler function.
    config
        Optional command config.

    Returns
    -------
    type[T]
        Decorated class.
    """
    effective_config = config or DEFAULT_CONFIG

    # Extract description from docstring if not provided
    op_description = effective_config.description or cls.__doc__ or f"Execute {operation_id}"
    op_description = op_description.strip().split("\n", maxsplit=1)[0].strip()

    # Extract group from operation_id (e.g., "jobs.list" -> "jobs")
    group = operation_id.split(".", maxsplit=1)[0]

    # Register operation
    register_operation(
        OperationSpec(
            operation_id=operation_id,
            name=cls.__name__,
            description=op_description,
            handler=handler,
            group=group,
            require_runtime=effective_config.require_runtime,
            require_gateway=effective_config.require_gateway,
            require_graph_runtime=effective_config.require_graph_runtime,
        )
    )

    # Generate __call__ method
    def generated_call(command_self: CommandInstance, *args: object, **kwargs: object) -> None:
        del args, kwargs
        _execute_command(
            command=command_self,
            operation_id=operation_id,
            handler=handler,
        )

    cls.__call__ = cast("Callable[..., object]", generated_call)
    return cls


def _decorate_new_style(
    cls: type[Command[Any]],
    operation_id: str,
    *,
    require_storage: bool,
    require_serving: bool,
) -> type[Command[Any]]:
    """Decorate with new Command[T] pattern.

    Parameters
    ----------
    cls
        Command class extending Command[T].
    operation_id
        Operation identifier.
    require_storage
        Whether storage access is required.
    require_serving
        Whether serving access is required.

    Returns
    -------
    type[CommandT]
        Decorated class.
    """
    # Set class attributes if not already defined
    if not hasattr(cls, "__operation_id__"):
        cls.__operation_id__ = operation_id
    if not hasattr(cls, "__require_storage__"):
        cls.__require_storage__ = require_storage
    if not hasattr(cls, "__require_serving__"):
        cls.__require_serving__ = require_serving

    # Extract description
    op_description = cls.__doc__ or f"Execute {operation_id}"
    op_description = op_description.strip().split("\n", maxsplit=1)[0].strip()

    group = operation_id.split(".", maxsplit=1)[0]

    # Create a wrapper handler for registry compatibility
    def command_handler(ctx: HandlerContext) -> CliResult[object]:
        # This is called if someone uses execute_operation() from registry
        # We need to convert HandlerContext to Deps and call execute()
        deps = deps_from_handler_context(ctx)
        cmd = _reconstruct_command(cls, ctx)
        return cmd.execute(deps)

    # Register operation
    register_operation(
        OperationSpec(
            operation_id=operation_id,
            name=cls.__name__,
            description=op_description,
            handler=command_handler,
            group=group,
            require_runtime=require_storage,
            require_gateway=require_storage,
            require_graph_runtime=False,
        )
    )

    # Generate __call__ method for CLI invocation
    def generated_call(command_self: Command[Any], *args: object, **kwargs: object) -> None:
        del args, kwargs
        _execute_new_command(
            command_self,
            require_storage=require_storage,
            require_serving=require_serving,
        )

    cls.__call__ = cast("Callable[..., object]", generated_call)
    return cls


def _reconstruct_command(
    cls: type[Command[Any]],
    ctx: HandlerContext,
) -> Command[Any]:
    """Reconstruct command instance from HandlerContext params.

    Parameters
    ----------
    cls
        Command class.
    ctx
        Handler context with params.

    Returns
    -------
    CommandType
        Reconstructed command instance.

    Raises
    ------
    TypeError
        If cls is not a dataclass Command type.
    """
    if not _is_dataclass_command_type(cls):
        msg = f"{cls.__name__} must be a dataclass Command"
        raise TypeError(msg)

    kwargs: dict[str, object] = {}
    dataclass_cls = cast("Any", cls)
    for fld in dataclasses.fields(dataclass_cls):
        if fld.name == "flags":
            # Create SharedFlags from context

            kwargs["flags"] = SharedFlags(
                output_format=ctx.output_format,
                verbose=ctx.verbosity,
                project_root=ctx.project_root,
            )
        elif fld.name in ctx._params:  # noqa: SLF001
            kwargs[fld.name] = ctx._params[fld.name]  # noqa: SLF001
        elif fld.default is not dataclasses.MISSING:
            kwargs[fld.name] = fld.default
        elif fld.default_factory is not dataclasses.MISSING:
            kwargs[fld.name] = fld.default_factory()

    return cast("Command[Any]", cls(**kwargs))


def _execute_new_command[T](
    command: Command[T],
    *,
    require_storage: bool,
    require_serving: bool,
) -> None:
    """Execute a new-style Command[T] using unified CommandContext.

    Parameters
    ----------
    command
        Command instance.
    require_storage
        Whether storage is required.
    require_serving
        Whether serving is required.
    """
    # Extract infrastructure from command
    infra = _extract_infrastructure(command)
    params = _extract_params(command)

    # Bootstrap CLI
    bootstrap_cli(verbosity=infra.verbosity)

    # Build CommandContext using the new unified builder
    builder = (
        CommandContextBuilder()
        .with_params(params)
        .with_output_format(infra.output_format)
        .with_verbosity(infra.verbosity)
        .with_operation_id(getattr(command, "__operation_id__", "unknown"))
    )

    if require_storage or require_serving:
        builder = builder.with_storage(db_path=infra.database_path)

    if require_serving:
        builder = builder.with_serving()

    # Execute with unified context
    with builder.build() as ctx:
        # Convert to Deps for backward compatibility with Command[T].execute()
        deps = deps_from_command_context(ctx)
        result = command.execute(deps)

    # Render result
    renderer = get_renderer(infra.output_format)
    exit_code = renderer.render_result(result)

    if exit_code != 0:
        sys.exit(exit_code)


def _execute_command[R](
    command: CommandInstance,
    operation_id: str,
    handler: Callable[[HandlerContext], CliResult[R]],
) -> None:
    """Execute a CLI command using unified CommandContext.

    Parameters
    ----------
    command
        Command dataclass instance.
    operation_id
        Operation identifier.
    handler
        Handler function.
    """
    # Extract standard infrastructure from SharedFlags mixin or inline fields
    infra = _extract_infrastructure(command)
    params = _extract_params(command)

    # Bootstrap CLI
    bootstrap_cli(verbosity=infra.verbosity)

    # Build unified CommandContext
    # Legacy handlers always need runtime/storage for backwards compatibility
    builder = (
        CommandContextBuilder()
        .with_params(params)
        .with_output_format(infra.output_format)
        .with_verbosity(infra.verbosity)
        .with_operation_id(operation_id)
        .with_runtime(project_root=infra.project_root)
        .with_storage(db_path=infra.database_path)
    )

    # Execute with unified context
    with builder.build() as ctx:
        # Convert to legacy HandlerContext for backward compatibility
        handler_ctx = handler_context_from_command_context(ctx)
        # Set additional legacy fields
        handler_ctx = HandlerContext(
            config=ctx.config,
            operation_id=operation_id,
            output_format=infra.output_format,
            verbosity=infra.verbosity,
            project_root=infra.project_root,
            database_path=infra.database_path,
            index_path=infra.index_path,
            _params=params,
        )
        try:
            with handler_ctx:
                result = handler(handler_ctx)
        except Exception:
            LOG.exception("Handler %s raised exception", operation_id)
            raise

    # Render result
    renderer = get_renderer(infra.output_format)
    exit_code = renderer.render_result(result)

    if exit_code != 0:
        sys.exit(exit_code)


@dataclass(frozen=True)
class _InfrastructureValues:
    """Extracted infrastructure values from command instance."""

    verbosity: int
    output_format: OutputFormat
    project_root: Path | None
    database_path: Path | None
    index_path: Path | None


def _extract_infrastructure(command: CommandInstance) -> _InfrastructureValues:
    """Extract infrastructure values from command, supporting SharedFlags mixin.

    Check for SharedFlags mixin first, then fall back to inline fields.

    Parameters
    ----------
    command
        Command dataclass instance.

    Returns
    -------
    _InfrastructureValues
        Extracted infrastructure values.
    """
    # Check for SharedFlags mixin
    flags = getattr(command, "flags", None)

    if flags is not None and hasattr(flags, "verbose"):
        # Extract from SharedFlags mixin
        verbosity = getattr(flags, "verbose", 0)
        output_format = _resolve_output_format_from_attrs(
            getattr(flags, "output_format", None),
            json_flag=getattr(flags, "json", False),
        )
        project_root = _convert_to_path(getattr(flags, "project_root", None))
    else:
        # Extract from inline fields
        verbosity = getattr(command, "verbose", 0)
        output_format = _get_output_format(command)
        project_root = _get_path_field(command, "project", "project_root")

    # These fields are always extracted from command directly
    database_path = _get_path_field(command, "db_path", "database_path")
    index_path = _get_path_field(command, "index_path")

    return _InfrastructureValues(
        verbosity=verbosity,
        output_format=output_format,
        project_root=project_root,
        database_path=database_path,
        index_path=index_path,
    )


def _resolve_output_format_from_attrs(
    output_format: OutputFormat | None,
    *,
    json_flag: bool,
) -> OutputFormat:
    """Resolve output format from attribute values.

    Parameters
    ----------
    output_format
        Explicit output format value.
    json_flag
        JSON flag value.

    Returns
    -------
    OutputFormat
        Resolved output format.
    """
    if json_flag:
        return OutputFormat.JSON
    if output_format is not None:
        if isinstance(output_format, OutputFormat):
            return output_format
        return OutputFormat(str(output_format))
    return OutputFormat.TEXT


def _convert_to_path(value: object) -> Path | None:
    """Convert value to Path if not None.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    Path | None
        Path value or None.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _get_output_format(command: CommandInstance) -> OutputFormat:
    """Get output format from command instance.

    Parameters
    ----------
    command
        Command dataclass instance.

    Returns
    -------
    OutputFormat
        Resolved output format.
    """
    # Check for explicit format field
    fmt = getattr(command, "output_format", None)
    if fmt is not None:
        if isinstance(fmt, OutputFormat):
            return fmt
        return OutputFormat(str(fmt))

    # Check for --json flag
    json_flag = getattr(command, "json", False)
    if json_flag:
        return OutputFormat.JSON

    return OutputFormat.TEXT


def _extract_params(command: CommandInstance) -> dict[str, object]:
    """Extract parameters from command dataclass fields.

    Infrastructure fields (output_format, verbose, etc.) are excluded.
    All other fields become handler parameters.

    Parameters
    ----------
    command
        Command dataclass instance.

    Returns
    -------
    dict[str, object]
        Parameter dictionary.
    """
    if not dataclasses.is_dataclass(command):
        return {}

    params: dict[str, object] = {}

    for field_info in dataclasses.fields(command):
        name = field_info.name

        # Skip infrastructure fields
        if name in _INFRASTRUCTURE_FIELDS:
            continue

        value = getattr(command, name)
        params[name] = value

    return params


def _get_path_field(command: CommandInstance, *field_names: str) -> Path | None:
    """Get Path value from first matching field.

    Parameters
    ----------
    command
        Command dataclass instance.
    field_names
        Field names to try in order.

    Returns
    -------
    Path | None
        Path value or None.
    """
    for name in field_names:
        if hasattr(command, name):
            value = getattr(command, name)
            if value is not None:
                if isinstance(value, Path):
                    return value
                return Path(str(value))
    return None


__all__ = [
    "CommandConfig",
    "cli_command",
    # Exported for testing
    "extract_infrastructure",
    "extract_params",
    "get_output_format",
    "get_path_field",
]


# Public aliases for testing (avoiding underscore prefix)
extract_infrastructure = _extract_infrastructure
extract_params = _extract_params
get_output_format = _get_output_format
get_path_field = _get_path_field
