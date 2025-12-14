"""Function span type definition.

This module provides the canonical FunctionSpan dataclass used across
the codebase for representing function metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionSpan:
    """Unified function span representation with optional URN.

    Attributes
    ----------
    goid
        Global object identifier (128-bit hash).
    rel_path
        Relative file path within the repository.
    qualname
        Fully qualified name of the function.
    start_line
        Starting line number (1-indexed).
    end_line
        Ending line number (1-indexed).
    urn
        Optional URN identifier. Populated when loaded via catalog.

    Examples
    --------
    >>> span = FunctionSpan(
    ...     goid=123456789,
    ...     rel_path="src/main.py",
    ...     qualname="module.MyClass.method",
    ...     start_line=10,
    ...     end_line=25,
    ... )
    >>> span.local_name
    'method'
    """

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    urn: str | None = None

    @property
    def local_name(self) -> str:
        """Extract the local (unqualified) function name.

        Returns
        -------
        str
            Local function name without module/class prefix.
        """
        return self.qualname.rsplit(".", maxsplit=1)[-1]

    @property
    def line_count(self) -> int:
        """Return the number of lines in the span.

        Returns
        -------
        int
            Number of lines (inclusive).
        """
        return self.end_line - self.start_line + 1

    def contains_line(self, line: int) -> bool:
        """Check if the span contains the given line.

        Parameters
        ----------
        line
            Line number to check.

        Returns
        -------
        bool
            True if line is within [start_line, end_line].
        """
        return self.start_line <= line <= self.end_line


__all__ = [
    "FunctionSpan",
]
