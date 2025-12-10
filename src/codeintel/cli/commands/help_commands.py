"""Enhanced help commands for operation discovery.

Provide commands to explore registered operations and view detailed
help with resource requirements.

Note: Phase 6 Migration - Schema command removed as new OperationSpec
no longer includes param_schema. Use CLI --help for parameter info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.introspection import get_help_renderer

help_commands_app = App(name="help", help="Get help on operations")


@help_commands_app.command(name="operation")
@dataclass
class HelpOperationCommand:
    """Show detailed help for a specific operation.

    Display operation description, resource requirements, and usage.
    Use this to understand how to invoke an operation correctly.
    """

    operation_id: Annotated[str, Parameter(help="Operation ID to describe")]

    def __call__(self) -> None:
        """Execute the help operation command.

        Raises
        ------
        SystemExit
            If the operation is not found.
        """
        renderer = get_help_renderer()
        if not renderer.render_operation_detail(self.operation_id):
            raise SystemExit(1)


@help_commands_app.command(name="list")
@dataclass
class HelpListCommand:
    """List all available operations.

    Display a table of all registered operations with their
    groups and descriptions.
    """

    by_group: Annotated[
        bool,
        Parameter(help="Group operations by group"),
    ] = False

    def __call__(self) -> None:
        """Execute the help list command."""
        renderer = get_help_renderer()
        renderer.render_operation_list(by_group=self.by_group)


@help_commands_app.command(name="search")
@dataclass
class HelpSearchCommand:
    """Search operations by name or description.

    Find operations matching a search query in their ID,
    name, or description text.
    """

    query: Annotated[str, Parameter(help="Search query")]

    def __call__(self) -> None:
        """Execute the help search command."""
        renderer = get_help_renderer()
        renderer.render_search_results(self.query)


__all__ = [
    "help_commands_app",
]
