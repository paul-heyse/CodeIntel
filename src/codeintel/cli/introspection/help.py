"""Enhanced help system with rich contextual output.

Provide detailed help for operations using introspection APIs,
including operation metadata and resource requirements.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codeintel.cli.execution.registry import OperationSpec
from codeintel.cli.introspection.discovery import (
    get_operation_info,
    list_all_operations,
    list_operations_by_group,
    search_operations,
)
from codeintel.cli.rendering import CODEINTEL_THEME

# Minimum number of parts in operation_id for group/action split (e.g., "jobs.list")
_MIN_OPERATION_ID_PARTS = 2


@dataclass
class HelpRenderer:
    """Render help content with rich formatting.

    Parameters
    ----------
    console
        Rich console for output.
    """

    console: Console

    def render_operation_detail(self, operation_id: str) -> bool:
        """Render detailed help for an operation.

        Parameters
        ----------
        operation_id
            Operation to describe.

        Returns
        -------
        bool
            True if operation found.
        """
        info = get_operation_info(operation_id)
        if info is None:
            self.console.print(f"[error]Operation not found: {operation_id}[/error]")
            return False

        # Header
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{info.operation_id}[/bold]\n\n{info.description}",
                title="Operation",
                border_style="cyan",
            )
        )

        # Metadata
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Name", info.name)
        table.add_row("Group", info.group)
        table.add_row("Requires Runtime", "Yes" if info.require_runtime else "No")
        table.add_row("Requires Gateway", "Yes" if info.require_gateway else "No")
        table.add_row("Requires Graph Runtime", "Yes" if info.require_graph_runtime else "No")
        if info.tags:
            table.add_row("Tags", ", ".join(info.tags))
        self.console.print(table)
        self.console.print()

        # Usage example
        self._render_usage_example(info)

        return True

    def _render_usage_example(self, info: OperationInfo) -> None:
        """Render usage example for an operation.

        Parameters
        ----------
        info
            Operation information.
        """
        self.console.print("[heading]Usage[/heading]")
        # Generate basic CLI example from operation_id
        parts = info.operation_id.split(".")
        if len(parts) >= _MIN_OPERATION_ID_PARTS:
            group, action = parts[0], parts[1]
            self.console.print(f"  [dim]$[/dim] codeintel {group} {action}")
        else:
            self.console.print(f"  [dim]$[/dim] codeintel op call {info.operation_id}")
        self.console.print()

    def render_operation_list(self, *, by_group: bool = False) -> None:
        """Render list of all operations.

        Parameters
        ----------
        by_group
            Group by operation group.
        """
        if by_group:
            groups = list_operations_by_group()
            if not groups:
                self.console.print("[dim]No operations registered[/dim]")
                return

            for group, op_ids in sorted(groups.items()):
                self.console.print(f"\n[heading]{group.upper()}[/heading]")
                for op_id in sorted(op_ids):
                    info = get_operation_info(op_id)
                    desc = info.description if info else ""
                    self.console.print(f"  [cyan]{op_id}[/cyan] - {desc}")
        else:
            operations = list_all_operations()
            if not operations:
                self.console.print("[dim]No operations registered[/dim]")
                return

            table = Table(title="Available Operations")
            table.add_column("Operation ID", style="cyan")
            table.add_column("Group")
            table.add_column("Description")
            for info in sorted(operations, key=lambda x: x.operation_id):
                table.add_row(info.operation_id, info.group, info.description)
            self.console.print(table)

    def render_search_results(self, query: str) -> None:
        """Render search results.

        Parameters
        ----------
        query
            Search query.
        """
        results = search_operations(query)
        if not results:
            self.console.print(f"[dim]No operations matching: {query}[/dim]")
            return

        self.console.print(f"\n[heading]Operations matching '{query}'[/heading]\n")
        for info in results:
            self.console.print(f"[cyan]{info.operation_id}[/cyan]")
            self.console.print(f"  {info.description}")
            self.console.print()


def get_help_renderer() -> HelpRenderer:
    """Get a help renderer instance.

    Returns
    -------
    HelpRenderer
        Configured renderer.
    """
    console = Console(theme=CODEINTEL_THEME)
    return HelpRenderer(console=console)


def render_help_text(
    operation_id: str,
    *,
    writer: TextIO = sys.stdout,
) -> bool:
    """Render help as plain text.

    Parameters
    ----------
    operation_id
        Operation to describe.
    writer
        Output writer.

    Returns
    -------
    bool
        True if operation found.
    """
    info = get_operation_info(operation_id)
    if info is None:
        writer.write(f"Operation not found: {operation_id}\n")
        return False

    writer.write(f"Operation: {info.operation_id}\n")
    writer.write(f"Name: {info.name}\n")
    writer.write(f"Group: {info.group}\n")
    writer.write(f"Description: {info.description}\n")
    writer.write(f"Requires Runtime: {'Yes' if info.require_runtime else 'No'}\n")
    writer.write(f"Requires Gateway: {'Yes' if info.require_gateway else 'No'}\n")
    writer.write(f"Requires Graph Runtime: {'Yes' if info.require_graph_runtime else 'No'}\n")
    if info.tags:
        writer.write(f"Tags: {', '.join(info.tags)}\n")

    # Generate basic usage example
    writer.write("\nUsage:\n")
    parts = info.operation_id.split(".")
    if len(parts) >= _MIN_OPERATION_ID_PARTS:
        group, action = parts[0], parts[1]
        writer.write(f"  $ codeintel {group} {action}\n")
    else:
        writer.write(f"  $ codeintel op call {info.operation_id}\n")

    return True


__all__ = [
    "HelpRenderer",
    "get_help_renderer",
    "render_help_text",
]
