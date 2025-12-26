"""Unified execution pipeline for CLI operations.

This package provides a single execution infrastructure that supports
handler-based operations and the Command[T] pattern.

The canonical `OperationSpec` is defined in `execution/registry.py` and
is used by `@cli_command` decorator and the introspection system.

Public surface
--------------
Use the registry-based APIs (`OperationSpec`, `register_operation`, `execute_operation`) as
the supported entry points for defining and running operations. All operations should:

1. Define handlers that accept `CommandContext` and return `CliResult`
2. Register via `OperationSpec` with required fields (operation_id, name, description, handler, group)
3. Use `@cli_command` decorator for command-line integration

For commands, prefer the Command[T] pattern:

1. Define command as a frozen dataclass extending `Command[T]`
2. Implement `execute(self, ctx: CommandContext) -> CliResult[T]`

Examples
--------
Register an operation:

>>> from codeintel.cli.execution import OperationSpec, register_operation
>>> from codeintel.cli.context import CommandContext
>>> from codeintel.cli.core import CliResult
>>>
>>> def my_handler(ctx: CommandContext) -> CliResult:
...     return CliResult.ok({"status": "done"})
>>>
>>> spec = OperationSpec(
...     operation_id="my.operation",
...     name="My Operation",
...     description="Does something",
...     handler=my_handler,
...     group="my",
... )
>>> register_operation(spec)
"""

from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    execute_operation,
    get_registry,
    register_operation,
    reset_registry,
)

__all__ = [
    "OperationRegistry",
    "OperationSpec",
    "execute_operation",
    "get_registry",
    "register_operation",
    "reset_registry",
]
