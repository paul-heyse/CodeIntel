"""Table rendering specifications.

This module defines the types for specifying table structure:

- ColumnSpec: Individual column configuration
- TableSpec: Complete table specification
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.cli.rendering.types import JustifyMethod


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a table column.

    Parameters
    ----------
    key
        Dictionary key to extract from row data.
    header
        Column header text.
    style
        Rich style for the column (e.g., "bold", "cyan").
    justify
        Text justification.
    width
        Fixed column width (None for auto).

    Examples
    --------
    >>> col = ColumnSpec("name", "Name", style="cyan")
    >>> col.key
    'name'
    """

    key: str
    header: str
    style: str | None = None
    justify: JustifyMethod = "left"
    width: int | None = None


@dataclass(frozen=True)
class TableSpec:
    """Specification for rendering a table.

    Parameters
    ----------
    columns
        Column specifications.
    title
        Optional table title.
    caption
        Optional table caption (footer).
    show_row_numbers
        Whether to show row numbers.
    empty_message
        Message when table has no rows.

    Examples
    --------
    >>> spec = TableSpec(
    ...     columns=(
    ...         ColumnSpec("id", "ID"),
    ...         ColumnSpec("name", "Name"),
    ...     ),
    ...     title="Users",
    ... )
    >>> spec.title
    'Users'
    """

    columns: tuple[ColumnSpec, ...]
    title: str | None = None
    caption: str | None = None
    show_row_numbers: bool = False
    empty_message: str = "No data."


__all__ = [
    "ColumnSpec",
    "TableSpec",
]
