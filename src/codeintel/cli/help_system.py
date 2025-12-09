"""Enhanced help system with rich contextual output.

Provide detailed help for operations using introspection APIs,
including parameter documentation, examples, and JSON schemas.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codeintel.cli.cli_render import CODEINTEL_THEME
from codeintel.cli.introspection import (
    get_operation_info,
    get_operation_schema,
    list_all_operations,
    list_operations_by_category,
    search_operations,
)


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
        table.add_row("Category", info.category)
        table.add_row("Progress", "Yes" if info.requires_progress else "No")
        table.add_row("Retryable", "Yes" if info.retryable else "No")
        self.console.print(table)
        self.console.print()

        # Parameters
        if info.parameters:
            self.console.print("[heading]Parameters[/heading]")
            param_table = Table()
            param_table.add_column("Name", style="cyan")
            param_table.add_column("Type")
            param_table.add_column("Required")
            for param in info.parameters:
                param_table.add_row(
                    str(param.get("name", "")),
                    str(param.get("type", "")),
                    "Yes" if param.get("required") else "No",
                )
            self.console.print(param_table)
            self.console.print()

        # Examples
        if info.examples:
            self.console.print("[heading]Examples[/heading]")
            for example in info.examples:
                self.console.print(f"  [dim]$[/dim] {example}")
            self.console.print()

        return True

    def render_operation_schema(self, operation_id: str) -> bool:
        """Render JSON Schema for operation parameters.

        Parameters
        ----------
        operation_id
            Operation to describe.

        Returns
        -------
        bool
            True if schema found or operation has no parameters.
        """
        info = get_operation_info(operation_id)
        if info is None:
            self.console.print(f"[error]Operation not found: {operation_id}[/error]")
            return False

        schema = get_operation_schema(operation_id)
        if schema is None:
            self.console.print("[dim]No parameters for this operation[/dim]")
            return True

        self.console.print(json.dumps(schema, indent=2))
        return True

    def render_operation_list(self, *, by_category: bool = False) -> None:
        """Render list of all operations.

        Parameters
        ----------
        by_category
            Group by category.
        """
        if by_category:
            categories = list_operations_by_category()
            if not categories:
                self.console.print("[dim]No operations registered[/dim]")
                return

            for category, op_ids in sorted(categories.items()):
                self.console.print(f"\n[heading]{category.upper()}[/heading]")
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
            table.add_column("Category")
            table.add_column("Description")
            for info in sorted(operations, key=lambda x: x.operation_id):
                table.add_row(info.operation_id, info.category, info.description)
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
    writer.write(f"Category: {info.category}\n")
    writer.write(f"Description: {info.description}\n")
    writer.write(f"Progress: {'Yes' if info.requires_progress else 'No'}\n")
    writer.write(f"Retryable: {'Yes' if info.retryable else 'No'}\n")

    if info.parameters:
        writer.write("\nParameters:\n")
        for param in info.parameters:
            name = param.get("name", "")
            ptype = param.get("type", "")
            required = "required" if param.get("required") else "optional"
            writer.write(f"  {name}: {ptype} ({required})\n")

    if info.examples:
        writer.write("\nExamples:\n")
        for example in info.examples:
            writer.write(f"  $ {example}\n")

    return True


__all__ = [
    "HelpRenderer",
    "get_help_renderer",
    "render_help_text",
]
