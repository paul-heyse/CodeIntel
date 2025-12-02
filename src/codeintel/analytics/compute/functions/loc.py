"""Pure computation of lines of code metrics.

This module provides functions to compute physical and logical lines
of code for functions. All functions are pure.

Examples
--------
>>> lines = [
...     "def example():",
...     "    # comment",
...     "    x = 1",
...     "    return x",
... ]
>>> loc = compute_loc(lines, start_line=1, end_line=4)
>>> loc.physical
4
>>> loc.logical
3
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinesOfCode:
    """Lines of code metrics for a code span.

    Attributes
    ----------
    physical
        Total number of lines (including blanks and comments).
    logical
        Number of non-blank, non-comment lines.
    blank
        Number of blank lines.
    comment
        Number of comment-only lines.
    """

    physical: int
    logical: int
    blank: int
    comment: int


def compute_loc(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> LinesOfCode:
    """Compute lines of code metrics for a span within source lines.

    Count physical and logical lines of code within a specified range.
    Logical LOC excludes blank lines and comment-only lines.

    Parameters
    ----------
    lines
        List of source lines (0-indexed internally, but line numbers are 1-indexed).
    start_line
        Starting line number (1-indexed, inclusive).
    end_line
        Ending line number (1-indexed, inclusive).

    Returns
    -------
    LinesOfCode
        Immutable container with line count metrics.

    Examples
    --------
    >>> source = ["def f():", "    # doc", "    pass", ""]
    >>> loc = compute_loc(source, 1, 4)
    >>> loc.physical
    4
    >>> loc.logical
    2
    """
    physical = end_line - start_line + 1
    logical = 0
    blank = 0
    comment = 0

    for line_num in range(start_line, end_line + 1):
        if 1 <= line_num <= len(lines):
            line = lines[line_num - 1]
            stripped = line.strip()

            if not stripped:
                blank += 1
            elif stripped.startswith("#"):
                comment += 1
            else:
                logical += 1

    return LinesOfCode(
        physical=physical,
        logical=logical,
        blank=blank,
        comment=comment,
    )


def count_logical_lines(lines: list[str]) -> int:
    """Count logical lines in a list of source lines.

    Convenience function that counts non-blank, non-comment lines
    across all provided lines.

    Parameters
    ----------
    lines
        List of source lines.

    Returns
    -------
    int
        Count of logical lines.

    Examples
    --------
    >>> count_logical_lines(["x = 1", "", "# comment", "y = 2"])
    2
    """
    result = compute_loc(lines, 1, len(lines))
    return result.logical


__all__ = [
    "LinesOfCode",
    "compute_loc",
    "count_logical_lines",
]
