"""Safe SQL primitives for validated table and column references.

This module provides value objects that validate SQL identifiers at construction
time, preventing SQL injection vulnerabilities when building dynamic queries.

The validation ensures identifiers follow standard SQL naming conventions:
- Start with a letter or underscore
- Contain only letters, digits, and underscores
- Are not excessively long (max 128 characters)

Examples
--------
>>> ref = SafeTableRef.from_key("core.ast_nodes")
>>> ref.full_name
'core.ast_nodes'
>>> ref.schema
'core'
>>> ref.table
'ast_nodes'

>>> col = SafeColumnRef("file_path")
>>> col.name
'file_path'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


class InvalidIdentifierError(ValueError):
    """Raise when an SQL identifier fails validation.

    Attributes
    ----------
    identifier
        The invalid identifier string.
    reason
        Description of why validation failed.
    """

    def __init__(self, identifier: str, reason: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        identifier
            The invalid identifier string.
        reason
            Description of why validation failed.
        """
        self.identifier = identifier
        self.reason = reason
        super().__init__(f"Invalid SQL identifier '{identifier}': {reason}")


@dataclass(frozen=True, slots=True)
class SafeColumnRef:
    """Validated column reference preventing SQL injection.

    This value object validates column names at construction time,
    ensuring they are safe to use in dynamic SQL queries.

    Attributes
    ----------
    name
        The validated column name.

    Raises
    ------
    InvalidIdentifierError
        If the column name fails validation.

    Examples
    --------
    >>> col = SafeColumnRef("file_path")
    >>> col.name
    'file_path'
    >>> str(col)
    'file_path'
    """

    name: str

    _VALID_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    _MAX_LENGTH: ClassVar[int] = 128

    def __post_init__(self) -> None:
        """Validate the column name after initialization."""
        _validate_identifier(self.name, "column", self._VALID_PATTERN, self._MAX_LENGTH)

    def __str__(self) -> str:
        """Return the column name as a string.

        Returns
        -------
        str
            The validated column name.
        """
        return self.name


@dataclass(frozen=True, slots=True)
class SafeTableRef:
    """Validated table reference preventing SQL injection.

    This value object validates schema and table names at construction time,
    ensuring they are safe to use in dynamic SQL queries.

    Attributes
    ----------
    schema
        The validated schema name.
    table
        The validated table name.

    Raises
    ------
    InvalidIdentifierError
        If either the schema or table name fails validation.

    Examples
    --------
    >>> ref = SafeTableRef(schema="core", table="ast_nodes")
    >>> ref.full_name
    'core.ast_nodes'
    >>> ref = SafeTableRef.from_key("analytics.metrics")
    >>> ref.schema
    'analytics'
    """

    schema: str
    table: str

    _VALID_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    _MAX_LENGTH: ClassVar[int] = 128

    def __post_init__(self) -> None:
        """Validate both schema and table names after initialization."""
        _validate_identifier(self.schema, "schema", self._VALID_PATTERN, self._MAX_LENGTH)
        _validate_identifier(self.table, "table", self._VALID_PATTERN, self._MAX_LENGTH)

    @classmethod
    def from_key(cls, table_key: str) -> SafeTableRef:
        """Parse a 'schema.table' key into a validated reference.

        Parameters
        ----------
        table_key
            Table key in 'schema.table' format.

        Returns
        -------
        SafeTableRef
            Validated table reference.

        Raises
        ------
        InvalidIdentifierError
            If the table key format is invalid or identifiers fail validation.

        Examples
        --------
        >>> ref = SafeTableRef.from_key("core.ast_nodes")
        >>> ref.schema
        'core'
        >>> ref.table
        'ast_nodes'
        """
        if "." not in table_key:
            raise InvalidIdentifierError(table_key, "table key must be in 'schema.table' format")
        parts = table_key.split(".", maxsplit=1)
        expected_parts = 2
        if len(parts) != expected_parts or not parts[0] or not parts[1]:
            raise InvalidIdentifierError(table_key, "table key must be in 'schema.table' format")
        return cls(schema=parts[0], table=parts[1])

    @property
    def full_name(self) -> str:
        """Return the fully qualified table name.

        Returns
        -------
        str
            The validated full table name in 'schema.table' format.
        """
        return f"{self.schema}.{self.table}"

    def __str__(self) -> str:
        """Return the full table name as a string.

        Returns
        -------
        str
            The validated full table name.
        """
        return self.full_name

    @staticmethod
    def column(name: str) -> SafeColumnRef:
        """Create a validated column reference.

        Parameters
        ----------
        name
            Column name to validate.

        Returns
        -------
        SafeColumnRef
            Validated column reference.
        """
        return SafeColumnRef(name)


def _validate_identifier(
    value: str,
    kind: str,
    pattern: re.Pattern[str],
    max_length: int,
) -> None:
    """Validate an SQL identifier.

    Parameters
    ----------
    value
        The identifier to validate.
    kind
        Kind of identifier (for error messages).
    pattern
        Regex pattern the identifier must match.
    max_length
        Maximum allowed length.

    Raises
    ------
    InvalidIdentifierError
        If validation fails.
    """
    if not value:
        raise InvalidIdentifierError(value, f"{kind} name cannot be empty")

    if len(value) > max_length:
        raise InvalidIdentifierError(value, f"{kind} name exceeds maximum length of {max_length}")

    if not pattern.match(value):
        raise InvalidIdentifierError(
            value,
            f"{kind} name must start with a letter or underscore "
            "and contain only letters, digits, and underscores",
        )


def validate_table_key(table_key: str) -> SafeTableRef:
    """Validate and parse a table key into a safe reference.

    This is a convenience function that wraps SafeTableRef.from_key().

    Parameters
    ----------
    table_key
        Table key in 'schema.table' format.

    Returns
    -------
    SafeTableRef
        Validated table reference.
    """
    return SafeTableRef.from_key(table_key)


def validate_column_name(name: str) -> SafeColumnRef:
    """Validate a column name and return a safe reference.

    This is a convenience function that wraps SafeColumnRef().

    Parameters
    ----------
    name
        Column name to validate.

    Returns
    -------
    SafeColumnRef
        Validated column reference.
    """
    return SafeColumnRef(name)


__all__ = [
    "InvalidIdentifierError",
    "SafeColumnRef",
    "SafeTableRef",
    "validate_column_name",
    "validate_table_key",
]
