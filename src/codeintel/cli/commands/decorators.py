"""Declarative command binding via @cli_command decorator.

This module provides the @cli_command decorator that eliminates boilerplate
from CLI command classes.

**Command[T] Pattern**:

- Command extends `Command[T]` from `cli.core.command`
- Implements `execute(self, ctx: CommandContext) -> CliResult[T]`
- Uses `require_storage` parameter
- Full end-to-end type safety

Handlers also use CommandContext directly via unified services.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeGuard, TypeVar, cast

from codeintel.cli.context import CommandContextBuilder
from codeintel.cli.core.command import Command
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.cli.options.shared_flags import SharedFlags
from codeintel.cli.rendering.service import get_renderer
from codeintel.cli.rendering.types import OutputFormat
from codeintel.observability import observe_operation, shutdown_observability

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.context import CommandContext
    from codeintel.cli.core import CliResult
    from codeintel.observability.cli import RunContext

LOG = logging.getLogger(__name__)


CommandInstance = object


class _DataclassCommand(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

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


DEFAULT_CONFIG = CommandConfig()


_INFRASTRUCTURE_FIELDS = frozenset(
    {
        "output_format",
        "verbose",
        "json",
        "project",
        "project_root",
        "db_path",
        "database_path",
        "index_path",
        "flags",
    }
)


def cli_command[T, R](
    operation_id: str,
    *,
    handler: Callable[[CommandContext], CliResult[R]] | None = None,
    config: CommandConfig | None = None,
    require_storage: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Decorate CLI command dataclasses with automatic execution.

    Support two patterns:

    **Handler Pattern** (handler= provided):
    Uses CommandContext-based handlers with CommandConfig.

    **Command[T] Pattern** (Command[T] subclass, no handler):
    Command implements execute(ctx) directly.

    Parameters
    ----------
    operation_id
        Unique operation identifier (e.g., "jobs.list").
    handler
        Handler function to invoke. If None, command must extend Command[T].
    config
        Command configuration (resource requirements, description).
        Defaults to requiring runtime and gateway.
    require_storage
        Whether command needs storage access via ctx.storage.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Class decorator.

    Examples
    --------
    Handler pattern:

    >>> from codeintel.cli.context import CommandContext
    >>> from codeintel.cli.core import CliResult
    >>> def my_handler(ctx: CommandContext) -> CliResult[dict[str, str]]:
    ...     return CliResult.ok({"status": "done"})
    >>>
    >>>
    >>>
    >>>

    Command[T] pattern:

    >>>
    >>>
    >>>
    >>>
    >>>
    >>>
    >>>
    """

    def decorator(cls: type[T]) -> type[T]:
        is_new_pattern = _is_command_subclass(cls)

        if is_new_pattern and handler is None:
            command_cls = cast("type[Command[Any]]", cls)
            decorated = _decorate_new_style(
                command_cls,
                operation_id,
                require_storage=require_storage,
            )
            return cast("type[T]", decorated)

        if handler is not None:
            return _decorate_handler_based(cls, operation_id, handler, config)

        msg = (
            f"@cli_command on {cls.__name__}: must provide handler= for handler "
            "pattern or extend Command[T] for command pattern"
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
        return any(
            getattr(base, "__name__", "") == "Command"
            and getattr(base, "__module__", "").startswith("codeintel.cli")
            for base in cls.__mro__
            if base is not object
        )
    except TypeError:
        return False


def _decorate_handler_based[T, R](
    cls: type[T],
    operation_id: str,
    handler: Callable[[CommandContext], CliResult[R]],
    config: CommandConfig | None,
) -> type[T]:
    """Decorate with handler-based pattern using CommandContext.

    Parameters
    ----------
    cls
        Command class.
    operation_id
        Operation identifier.
    handler
        Handler function receiving CommandContext.
    config
        Optional command config.

    Returns
    -------
    type[T]
        Decorated class.
    """
    effective_config = config or DEFAULT_CONFIG

    op_description = effective_config.description or cls.__doc__ or f"Execute {operation_id}"
    op_description = op_description.strip().split("\n", maxsplit=1)[0].strip()

    group = operation_id.split(".", maxsplit=1)[0]

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

    def generated_call(command_self: CommandInstance, *args: object, **kwargs: object) -> None:
        del args, kwargs
        _execute_handler_command(
            command=command_self,
            operation_id=operation_id,
            handler=handler,
            config=effective_config,
        )

    cls.__call__ = cast("Callable[..., object]", generated_call)
    return cls


def _decorate_new_style(
    cls: type[Command[Any]],
    operation_id: str,
    *,
    require_storage: bool,
) -> type[Command[Any]]:
    """Decorate with Command[T] pattern using CommandContext.

    Parameters
    ----------
    cls
        Command class extending Command[T].
    operation_id
        Operation identifier.
    require_storage
        Whether storage access is required.

    Returns
    -------
    type[CommandT]
        Decorated class.
    """
    if not hasattr(cls, "__operation_id__"):
        cls.__operation_id__ = operation_id
    if not hasattr(cls, "__require_storage__"):
        cls.__require_storage__ = require_storage

    op_description = cls.__doc__ or f"Execute {operation_id}"
    op_description = op_description.strip().split("\n", maxsplit=1)[0].strip()

    group = operation_id.split(".", maxsplit=1)[0]

    def command_handler(ctx: CommandContext) -> CliResult[object]:
        cmd = _reconstruct_command_from_context(cls, ctx)
        return cmd.execute(ctx)

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

    def generated_call(command_self: Command[Any], *args: object, **kwargs: object) -> None:
        del args, kwargs
        _execute_new_command(
            command_self,
            require_storage=require_storage,
        )

    cls.__call__ = cast("Callable[..., object]", generated_call)
    return cls


def _reconstruct_command_from_context(
    cls: type[Command[Any]],
    ctx: CommandContext,
) -> Command[Any]:
    """Reconstruct command instance from CommandContext params.

    Parameters
    ----------
    cls
        Command class.
    ctx
        Command context with params.

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
    for fld in dataclasses.fields(cls):
        if fld.name == "flags":
            flags = _build_flags_from_context(fld, ctx)
            if flags is not None:
                kwargs["flags"] = flags
                continue

        if fld.name in ctx.params.raw:
            kwargs[fld.name] = ctx.params.raw[fld.name]
        elif fld.default is not dataclasses.MISSING:
            kwargs[fld.name] = fld.default
        elif fld.default_factory is not dataclasses.MISSING:
            kwargs[fld.name] = fld.default_factory()

    return cast("Command[Any]", cls(**kwargs))


def _build_flags_from_context(
    fld: dataclasses.Field[object],
    ctx: CommandContext,
) -> SharedFlags | None:
    default_flags: object | None
    if fld.default_factory is not dataclasses.MISSING:
        default_flags = fld.default_factory()
    elif fld.default is not dataclasses.MISSING:
        default_flags = fld.default
    else:
        return None

    if not dataclasses.is_dataclass(default_flags):
        return None

    flags = cast("SharedFlags", default_flags)

    field_names = {field.name for field in dataclasses.fields(default_flags)}
    replace_kwargs: dict[str, object] = {}
    if "output_format" in field_names:
        replace_kwargs["output_format"] = ctx.output_format
    if "json" in field_names:
        replace_kwargs["json"] = ctx.output_format == OutputFormat.JSON
    if "verbose" in field_names:
        replace_kwargs["verbose"] = ctx.verbosity
    if "project_root" in field_names:
        replace_kwargs["project_root"] = ctx.runtime.root if ctx.runtime else None
    if "run_context" in field_names:
        replace_kwargs["run_context"] = ctx.run_context

    if not replace_kwargs:
        return flags
    return _replace_dataclass_instance(flags, replace_kwargs)


def _replace_dataclass_instance[TDataclass](
    instance: TDataclass,
    updates: Mapping[str, object],
) -> TDataclass:
    field_map = getattr(instance, "__dataclass_fields__", None)
    if not isinstance(field_map, dict):
        msg = "Expected a dataclass instance for flag updates"
        raise TypeError(msg)
    values = {name: getattr(instance, name) for name in field_map}
    values.update(updates)
    instance_type = cast("type[TDataclass]", type(instance))
    return instance_type(**values)


def _execute_new_command[T](
    command: Command[T],
    *,
    require_storage: bool,
) -> None:
    """Execute a Command[T] using unified CommandContext.

    Parameters
    ----------
    command
        Command instance.
    require_storage
        Whether storage is required.
    """
    infra = _extract_infrastructure(command)
    params = _extract_params(command)

    bootstrap_cli(verbosity=infra.verbosity)

    builder = (
        CommandContextBuilder()
        .with_params(params)
        .with_output_format(infra.output_format)
        .with_verbosity(infra.verbosity)
        .with_operation_id(getattr(command, "__operation_id__", "unknown"))
        .with_run_context(infra.run_context)
    )

    if require_storage:
        builder = builder.with_storage(db_path=infra.database_path)

    with builder.build() as ctx:
        try:
            with observe_operation(
                component="cli",
                operation=getattr(command, "__operation_id__", "unknown"),
                attributes={"codeintel.output_format": str(infra.output_format)},
            ):
                result = command.execute(ctx)
        except Exception:
            LOG.exception(
                "Command %s raised exception",
                getattr(command, "__operation_id__", "unknown"),
            )
            raise
        finally:
            shutdown_observability()

    renderer = get_renderer(infra.output_format)
    exit_code = renderer.render_result(result)

    if exit_code != 0:
        sys.exit(exit_code)


def _execute_handler_command[R](
    command: CommandInstance,
    operation_id: str,
    handler: Callable[[CommandContext], CliResult[R]],
    config: CommandConfig,
) -> None:
    """Execute a CLI handler command using unified CommandContext.

    Parameters
    ----------
    command
        Command dataclass instance.
    operation_id
        Operation identifier.
    handler
        Handler function receiving CommandContext.
    config
        Command configuration.
    """
    infra = _extract_infrastructure(command)
    params = _extract_params(command)

    bootstrap_cli(verbosity=infra.verbosity)

    builder = (
        CommandContextBuilder()
        .with_params(params)
        .with_output_format(infra.output_format)
        .with_verbosity(infra.verbosity)
        .with_operation_id(operation_id)
        .with_run_context(infra.run_context)
    )

    if config.require_runtime:
        builder = builder.with_runtime(project_root=infra.project_root)

    if config.require_gateway:
        builder = builder.with_storage(db_path=infra.database_path)

    with builder.build() as ctx:
        try:
            with observe_operation(
                component="cli",
                operation=operation_id,
                attributes={"codeintel.output_format": str(infra.output_format)},
            ):
                result = handler(ctx)
        except Exception:
            LOG.exception("Handler %s raised exception", operation_id)
            raise
        finally:
            shutdown_observability()

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
    run_context: RunContext | None


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
    flags = getattr(command, "flags", None)

    if flags is not None and hasattr(flags, "verbose"):
        verbosity = getattr(flags, "verbose", 0)
        output_format = _resolve_output_format_from_attrs(
            getattr(flags, "output_format", None),
            json_flag=getattr(flags, "json", False),
        )
        project_root = _convert_to_path(getattr(flags, "project_root", None))
        run_context = getattr(flags, "run_context", None)
    else:
        verbosity = getattr(command, "verbose", 0)
        output_format = _get_output_format(command)
        project_root = _get_path_field(command, "project", "project_root")
        run_context = getattr(command, "run_context", None)

    database_path = _get_path_field(command, "db_path", "database_path")
    index_path = _get_path_field(command, "index_path")

    return _InfrastructureValues(
        verbosity=verbosity,
        output_format=output_format,
        project_root=project_root,
        database_path=database_path,
        index_path=index_path,
        run_context=run_context,
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
    fmt = getattr(command, "output_format", None)
    if fmt is not None:
        if isinstance(fmt, OutputFormat):
            return fmt
        return OutputFormat(str(fmt))

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
    "extract_infrastructure",
    "extract_params",
    "get_output_format",
    "get_path_field",
]


extract_infrastructure = _extract_infrastructure
extract_params = _extract_params
get_output_format = _get_output_format
get_path_field = _get_path_field
