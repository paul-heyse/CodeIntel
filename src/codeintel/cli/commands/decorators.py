"""Declarative command binding via @cli_command decorator.

This module provides the @cli_command decorator that eliminates boilerplate
from CLI command classes. Instead of manually implementing __call__, the
decorator generates it based on:

- Handler function to invoke
- Resource requirements
- Command dataclass fields
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.service import get_renderer
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

LOG = logging.getLogger(__name__)

# Protocol for command dataclass instances
CommandInstance = object  # Runtime type for dataclass instances


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
        "commit",
        "build_dir",
        "repo_root",
        "index_path",
        "flags",  # SharedFlags mixin field
    }
)


def cli_command[T, R](
    operation_id: str,
    *,
    handler: Callable[[HandlerContext], CliResult[R]],
    config: CommandConfig | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorate CLI command dataclasses with automatic execution.

    Generate a __call__ method that handles all CLI infrastructure:

    1. Bootstrap CLI (logging, config)
    2. Extract parameters from dataclass fields
    3. Create HandlerContext
    4. Invoke handler
    5. Render result
    6. Handle exit code

    Also registers the operation with the global OperationRegistry (NEW registry).

    Parameters
    ----------
    operation_id
        Unique operation identifier (e.g., "jobs.list").
    handler
        Handler function to invoke.
    config
        Optional command configuration (resource requirements, description).
        Defaults to requiring runtime and gateway.

    Returns
    -------
    Callable[[type[T]], type[T]]
        Class decorator.

    Examples
    --------
    Basic usage with default config (requires runtime and gateway):

    >>> from codeintel.cli.handlers.context import HandlerContext
    >>> from codeintel.cli.core import CliResult
    >>> def my_handler(ctx: HandlerContext) -> CliResult[dict[str, str]]:
    ...     return CliResult.ok({"status": "done"})
    >>> # @cli_command("my.op", handler=my_handler)
    >>> # @dataclass
    >>> # class MyCommand:
    >>> #     name: str = "default"

    With custom config (no runtime required):

    >>> cfg = CommandConfig(require_runtime=False, require_gateway=False)
    >>> # @cli_command("jobs.list", handler=list_handler, config=cfg)
    >>> # @dataclass
    >>> # class ListCommand:
    >>> #     limit: int = 20
    """
    effective_config = config or DEFAULT_CONFIG

    def decorator(cls: type[T]) -> type[T]:
        # Extract description from docstring if not provided
        op_description = effective_config.description or cls.__doc__ or f"Execute {operation_id}"
        # Clean up multi-line docstrings - get first line only
        op_description = op_description.strip().split("\n", maxsplit=1)[0].strip()

        # Extract group from operation_id (e.g., "jobs.list" -> "jobs")
        group = operation_id.split(".", maxsplit=1)[0]

        # Register operation with NEW registry
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
        def generated_call(command_self: CommandInstance) -> None:
            _execute_command(
                command=command_self,
                operation_id=operation_id,
                handler=handler,
            )

        # Attach to class
        cls.__call__ = generated_call  # type: ignore[attr-defined]

        return cls

    return decorator


def _execute_command[R](
    command: CommandInstance,
    operation_id: str,
    handler: Callable[[HandlerContext], CliResult[R]],
) -> None:
    """Execute a CLI command.

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

    # Bootstrap CLI
    cli_config = bootstrap_cli(verbosity=infra.verbosity)

    # Extract parameters
    params = _extract_params(command)

    # Create context
    ctx = HandlerContext(
        config=cli_config,
        operation_id=operation_id,
        output_format=infra.output_format,
        verbosity=infra.verbosity,
        project_root=infra.project_root,
        database_path=infra.database_path,
        index_path=infra.index_path,
        _params=params,
    )

    # Execute handler with context manager for cleanup
    try:
        with ctx:
            result = handler(ctx)
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
