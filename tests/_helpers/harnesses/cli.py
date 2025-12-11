"""CLI handler test harnesses.

This module provides test harnesses for CLI handler testing following
the hexagonal architecture pattern. The harnesses wrap CliTestContext
and provide convenient methods for executing handlers and asserting results.

The design mirrors the analytics plugin harness pattern to ensure
consistent testing approaches across the codebase.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.cli_context import create_cli_test_context
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK, SUBSYSTEM_PACK
from tests._helpers.seeds.cli import (
    CLI_CORE_PACK,
    GRAPH_HANDLER_PACK,
    STORAGE_PROFILE_PACK,
    SUBSYSTEM_HANDLER_PACK,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from codeintel.cli.context import CommandContext
    from codeintel.cli.core import CliResult
    from tests._helpers.cli_context import CliTestContext
    from tests._helpers.context import SeedPack


@dataclass
class CliHandlerHarness:
    """Lightweight harness for executing CLI handlers.

    Wrap CliTestContext and provide methods for handler execution
    and result access.

    Attributes
    ----------
    ctx : CliTestContext
        Underlying CLI test context.

    Examples
    --------
    >>> from tests._helpers.harnesses.cli import cli_handler_harness
    >>>
    >>> def test_handler(tmp_path):  # doctest: +SKIP
    ...     with cli_handler_harness(tmp_path) as harness:
    ...         result = harness.execute(my_handler, {"key": "value"})
    ...         assert result.success
    """

    ctx: CliTestContext

    def close(self) -> None:
        """Close the underlying CliTestContext."""
        self.ctx.close()

    def execute[T](
        self,
        handler: Callable[[CommandContext], CliResult[T]],
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> CliResult[T]:
        """Execute a handler function with the given parameters.

        Build a CommandContext and invoke the handler, returning the result.

        Parameters
        ----------
        handler
            Handler function to execute.
        params
            Handler parameters dictionary.
        operation_id
            Optional operation ID override.

        Returns
        -------
        CliResult[T]
            Result from the handler execution.

        Examples
        --------
        >>> result = harness.execute(my_handler, {"name": "test"})  # doctest: +SKIP
        >>> assert result.success
        """
        with self.ctx.command_context(params, operation_id=operation_id) as cmd_ctx:
            return handler(cmd_ctx)

    def execute_with_context[T](
        self,
        handler: Callable[[CommandContext], CliResult[T]],
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> tuple[CliResult[T], CommandContext]:
        """Execute a handler and return both result and context.

        Useful when tests need to inspect the command context after execution.

        Parameters
        ----------
        handler
            Handler function to execute.
        params
            Handler parameters dictionary.
        operation_id
            Optional operation ID override.

        Returns
        -------
        tuple[CliResult[T], CommandContext]
            Result and the command context used.
        """
        cmd_ctx = self.ctx.build_command_context(params, operation_id=operation_id)
        result = handler(cmd_ctx)
        return result, cmd_ctx

    def query_count(self, table: str, where: str | None = None) -> int:
        """Count rows in a table.

        Parameters
        ----------
        table
            Table name (schema.table format).
        where
            Optional WHERE clause.

        Returns
        -------
        int
            Number of rows.
        """
        return self.ctx.query_count(table, where)


def _apply_packs(ctx: CliTestContext, packs: tuple[SeedPack, ...]) -> None:
    """Apply seed packs to a CLI test context."""
    if packs:
        ctx.require(*packs)


@contextmanager
def cli_handler_harness(
    tmp_path: Path,
    *packs: SeedPack,
) -> Iterator[CliHandlerHarness]:
    """Create a CLI handler harness with specified seed packs.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    packs
        Seed packs to apply.

    Yields
    ------
    CliHandlerHarness
        Configured harness with seeds applied.

    Examples
    --------
    >>> from tests._helpers.harnesses.cli import cli_handler_harness
    >>> from tests._helpers.seeds import CORE_PACK
    >>>
    >>> def test_handler(tmp_path):  # doctest: +SKIP
    ...     with cli_handler_harness(tmp_path, CORE_PACK) as harness:
    ...         result = harness.execute(my_handler, {})
    """
    ctx = create_cli_test_context(tmp_path)
    _apply_packs(ctx, packs)
    harness = CliHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


@contextmanager
def core_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Create a handler harness with core seeds applied.

    Convenience factory for handlers that only need basic core data.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    CliHandlerHarness
        Harness with CORE_PACK and CLI_CORE_PACK seeds.
    """
    with cli_handler_harness(tmp_path, CORE_PACK, CLI_CORE_PACK) as harness:
        yield harness


@contextmanager
def storage_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Create a handler harness for storage handler tests.

    Seed with storage profile data needed for macro validation and
    profile operations.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    CliHandlerHarness
        Harness seeded for storage handler tests.
    """
    with cli_handler_harness(tmp_path, CORE_PACK, STORAGE_PROFILE_PACK) as harness:
        yield harness


@contextmanager
def graph_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Create a handler harness for graph handler tests.

    Seed with graph data needed for graph plugin operations.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    CliHandlerHarness
        Harness seeded for graph handler tests.
    """
    with cli_handler_harness(tmp_path, CORE_PACK, GRAPH_PACK, GRAPH_HANDLER_PACK) as harness:
        yield harness


@contextmanager
def subsystem_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Create a handler harness for subsystem handler tests.

    Seed with subsystem data needed for subsystem operations.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    CliHandlerHarness
        Harness seeded for subsystem handler tests.
    """
    with cli_handler_harness(
        tmp_path, CORE_PACK, SUBSYSTEM_PACK, SUBSYSTEM_HANDLER_PACK
    ) as harness:
        yield harness


@contextmanager
def ops_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Create a handler harness for ops handler tests.

    Seed with core data needed for operation listing and execution.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.

    Yields
    ------
    CliHandlerHarness
        Harness seeded for ops handler tests.
    """
    with cli_handler_harness(tmp_path, CORE_PACK, CLI_CORE_PACK) as harness:
        yield harness


__all__ = [
    "CliHandlerHarness",
    "cli_handler_harness",
    "core_handler_harness",
    "graph_handler_harness",
    "ops_handler_harness",
    "storage_handler_harness",
    "subsystem_handler_harness",
]
