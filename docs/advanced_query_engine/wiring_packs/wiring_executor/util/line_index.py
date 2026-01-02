from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from typing import Iterable, Tuple


@dataclass
class LineIndex:
    """Byte-based line/column mapping for UTF-8 source bytes.

    - Line numbers are 1-indexed.
    - Columns are 0-indexed (byte columns, not necessarily char columns).
    """

    src: bytes
    line_starts: list[int]

    @staticmethod
    def build(src: bytes) -> "LineIndex":
        starts = [0]
        # Treat '\n' as line separator; keep it simple and byte-based.
        for i, b in enumerate(src):
            if b == 0x0A:  # '\n'
                starts.append(i + 1)
        return LineIndex(src=src, line_starts=starts)

    def line_col(self, byte_offset: int) -> Tuple[int, int]:
        if byte_offset < 0:
            byte_offset = 0
        if byte_offset > len(self.src):
            byte_offset = len(self.src)

        # bisect_right returns insertion point; subtract 1 to get containing line start index.
        li = bisect_right(self.line_starts, byte_offset) - 1
        line_start = self.line_starts[li]
        line_no = li + 1  # 1-indexed
        col = byte_offset - line_start
        return line_no, col

    def span_to_range(self, start: int, end: int) -> dict:
        sl, sc = self.line_col(start)
        el, ec = self.line_col(end)
        return {
            "start_line": sl,
            "start_col": sc,
            "end_line": el,
            "end_col": ec,
        }

    def line_start_byte(self, line_no: int) -> int:
        if line_no <= 1:
            return 0
        idx = line_no - 1
        if idx >= len(self.line_starts):
            return len(self.src)
        return self.line_starts[idx]

    def extract_lines_around_span(self, start: int, end: int, before: int = 1, after: int = 1) -> dict:
        sl, _ = self.line_col(start)
        el, _ = self.line_col(end)
        first = max(1, sl - before)
        last = min(len(self.line_starts), el + after)

        lines: list[str] = []
        for ln in range(first, last + 1):
            a = self.line_start_byte(ln)
            b = self.line_start_byte(ln + 1)
            # include line text (strip trailing newline for readability)
            txt = self.src[a:b].decode("utf-8", errors="replace")
            lines.append(txt.rstrip("\n"))
        return {
            "first_line": first,
            "last_line": last,
            "lines": lines,
        }
