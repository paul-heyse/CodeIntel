"""Byte-based line/column indexing utilities."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

NEWLINE_BYTE = 0x0A


@dataclass(frozen=True)
class LineIndex:
    """Map byte offsets to line/column coordinates for UTF-8 bytes.

    Notes
    -----
    Line numbers are 1-indexed; columns are 0-indexed and byte-based.
    """

    source: bytes
    line_starts: list[int]

    @staticmethod
    def build(source: bytes) -> LineIndex:
        """Build a line index for the provided UTF-8 bytes.

        Parameters
        ----------
        source:
            UTF-8 encoded source bytes.

        Returns
        -------
        LineIndex
            Line index for the provided source.
        """
        starts = [0]
        for offset, byte in enumerate(source):
            if byte == NEWLINE_BYTE:
                starts.append(offset + 1)
        return LineIndex(source=source, line_starts=starts)

    def line_col(self, byte_offset: int) -> tuple[int, int]:
        """Return (line, col) for a byte offset.

        Parameters
        ----------
        byte_offset:
            Byte offset into the source.

        Returns
        -------
        tuple[int, int]
            Line number (1-indexed) and column (0-indexed).
        """
        normalized = min(max(byte_offset, 0), len(self.source))
        idx = bisect_right(self.line_starts, normalized) - 1
        line_start = self.line_starts[idx]
        line_no = idx + 1
        column = normalized - line_start
        return line_no, column

    def span_to_range(self, start: int, end: int) -> dict[str, int]:
        """Return a line/column range dict for byte offsets.

        Parameters
        ----------
        start:
            Starting byte offset.
        end:
            Ending byte offset.

        Returns
        -------
        dict[str, int]
            Mapping of start/end line and column values.
        """
        start_line, start_col = self.line_col(start)
        end_line, end_col = self.line_col(end)
        return {
            "start_line": start_line,
            "start_col": start_col,
            "end_line": end_line,
            "end_col": end_col,
        }

    def line_start_byte(self, line_no: int) -> int:
        """Return the starting byte offset for a line number.

        Parameters
        ----------
        line_no:
            Line number (1-indexed).

        Returns
        -------
        int
            Byte offset for the start of the line.
        """
        if line_no <= 1:
            return 0
        idx = line_no - 1
        if idx >= len(self.line_starts):
            return len(self.source)
        return self.line_starts[idx]

    def extract_lines_around(
        self,
        *,
        start: int,
        end: int,
        before: int = 1,
        after: int = 1,
    ) -> dict[str, object]:
        """Extract context lines around a byte span.

        Parameters
        ----------
        start:
            Starting byte offset.
        end:
            Ending byte offset.
        before:
            Number of lines of context before the span.
        after:
            Number of lines of context after the span.

        Returns
        -------
        dict[str, object]
            Mapping with line range metadata and extracted lines.
        """
        start_line, _ = self.line_col(start)
        end_line, _ = self.line_col(end)
        first = max(1, start_line - before)
        last = min(len(self.line_starts), end_line + after)

        lines: list[str] = []
        for line_no in range(first, last + 1):
            begin = self.line_start_byte(line_no)
            finish = self.line_start_byte(line_no + 1)
            text = self.source[begin:finish].decode("utf-8", errors="replace")
            lines.append(text.rstrip("\n"))

        return {"first_line": first, "last_line": last, "lines": lines}


__all__ = ["LineIndex"]
