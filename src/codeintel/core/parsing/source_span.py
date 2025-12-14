"""Source span type definition.

This module provides the canonical SourceSpan dataclass for representing
code locations across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SourceSpan:
    """Source span with file path and column information.

    Represents a contiguous span of source code with precise location
    including both line and column boundaries.

    Attributes
    ----------
    path
        Path to the source file.
    start_line
        Starting line number (1-indexed).
    start_col
        Starting column number (0-indexed).
    end_line
        Ending line number (1-indexed).
    end_col
        Ending column number (0-indexed).

    Examples
    --------
    >>> from pathlib import Path
    >>> span = SourceSpan(Path("src/main.py"), 10, 0, 25, 40)
    >>> span.line_count
    16
    """

    path: Path
    start_line: int
    start_col: int
    end_line: int
    end_col: int

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

    def overlaps(self, other: SourceSpan) -> bool:
        """Check if this span overlaps with another span.

        Parameters
        ----------
        other
            Another source span.

        Returns
        -------
        bool
            True if the spans overlap.
        """
        if self.path != other.path:
            return False
        return not (self.end_line < other.start_line or other.end_line < self.start_line)


__all__ = [
    "SourceSpan",
]
