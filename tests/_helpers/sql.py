"""SQL-related helpers for test assertions.

These helpers validate identifiers to prevent unsafe interpolation
when constructing lightweight SQL statements in tests.
"""

from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def validate_identifier(name: str, *, kind: str = "identifier") -> str:
    """Validate that a string is a safe SQL identifier.

    Parameters
    ----------
    name
        Identifier to validate (schema.table or column name).
    kind
        Description for error messages.

    Returns
    -------
    str
        The original identifier if valid.

    Raises
    ------
    ValueError
        If the identifier contains invalid characters.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        msg = f"Invalid {kind} identifier: {name!r}"
        raise ValueError(msg)
    return name


__all__ = ["validate_identifier"]
