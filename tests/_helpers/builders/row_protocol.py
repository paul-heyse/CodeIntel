"""Generic row insertion protocol for test data seeding.

This module provides a protocol-based approach to DuckDB row insertion,
eliminating repetitive insert_* function definitions by using class-level
metadata to drive generic insertion logic.

Design Notes
------------
- InsertableRow is a Protocol that row dataclasses implement
- Each row class declares its target table and column names as ClassVars
- The generic insert_rows() function uses this metadata for insertion
- Complex rows (with optional fields, special serialization) can override
  insertion behavior via custom insert methods
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# Regex for validating SQL identifiers (table/column names)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _validate_identifier(name: str, kind: str) -> str:
    """Validate that a string is a safe SQL identifier.

    Parameters
    ----------
    name
        The identifier to validate.
    kind
        Description of identifier type for error messages.

    Returns
    -------
    str
        The validated identifier.

    Raises
    ------
    ValueError
        If the identifier contains invalid characters.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        msg = f"Invalid {kind} identifier: {name!r}"
        raise ValueError(msg)
    return name


@runtime_checkable
class InsertableRow(Protocol):
    """Protocol for rows that can be inserted into DuckDB tables.

    Row dataclasses implement this protocol by declaring class variables
    for table name and column names, plus a to_tuple() method that returns
    field values in column order.

    Attributes
    ----------
    __table__ : ClassVar[str]
        Fully qualified table name (e.g., "core.modules").
    __columns__ : ClassVar[tuple[str, ...]]
        Column names in insertion order.

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class ModuleRow:
    ...     __table__: ClassVar[str] = "core.modules"
    ...     __columns__: ClassVar[tuple[str, ...]] = ("module", "path")
    ...     module: str
    ...     path: str
    ...
    ...     def to_tuple(self) -> tuple[str, str]:
    ...         return (self.module, self.path)
    """

    __table__: ClassVar[str]
    __columns__: ClassVar[tuple[str, ...]]

    def to_tuple(self) -> tuple[object, ...]:
        """Convert row to tuple of values in column order.

        Returns
        -------
        tuple[object, ...]
            Field values matching __columns__ order.
        """
        ...


def insert_rows(
    gateway: StorageGateway,
    rows: Iterable[InsertableRow],
) -> int:
    """Insert rows into their declared table using generic SQL generation.

    Generate and execute an INSERT statement based on the row class's
    __table__ and __columns__ metadata. This eliminates the need for
    individual insert_* functions for each row type.

    Parameters
    ----------
    gateway
        Storage gateway providing database connection.
    rows
        Iterable of InsertableRow instances to insert. All rows must be
        of the same concrete type.

    Returns
    -------
    int
        Number of rows inserted. Returns 0 if rows is empty.

    Examples
    --------
    >>> from tests._helpers.builders import ModuleRow
    >>> rows = [
    ...     ModuleRow(module="pkg.mod", path="pkg/mod.py", repo="r", commit="c"),
    ... ]
    >>> count = insert_rows(gateway, rows)
    """
    # Materialize to list if needed for length check and iteration
    row_list: Sequence[InsertableRow] = rows if isinstance(rows, Sequence) else list(rows)

    if not row_list:
        return 0

    # Extract metadata from first row's class
    sample = row_list[0]
    row_type = type(sample)

    # Access class variables - these are defined on the class, not instance
    table: str = row_type.__table__  # type: ignore[attr-defined]
    columns: tuple[str, ...] = row_type.__columns__  # type: ignore[attr-defined]

    # Validate identifiers to prevent SQL injection
    _validate_identifier(table, "table")
    validated_columns = [_validate_identifier(col, "column") for col in columns]

    # Build parameterized SQL by joining validated parts (avoids S608 false positive)
    col_names = ", ".join(validated_columns)
    placeholders = ", ".join("?" for _ in validated_columns)
    sql = " ".join(["INSERT INTO", table, "(", col_names, ")", "VALUES", "(", placeholders, ")"])

    # Execute batch insert
    gateway.con.executemany(sql, [r.to_tuple() for r in row_list])
    return len(row_list)


__all__ = ["InsertableRow", "insert_rows"]
