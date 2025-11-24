"""Shared AST utilities for capture and span lookup."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AstSpanIndex:
    """Index AST nodes by (start_line, end_line) spans for quick lookup."""

    node_map: dict[tuple[int, int], ast.AST]

    @classmethod
    def from_tree(
        cls,
        tree: ast.AST,
        kinds: tuple[type[ast.AST], ...],
    ) -> AstSpanIndex:
        """
        Build an index for the given AST and target node kinds.

        Returns
        -------
        AstSpanIndex
            Span index mapping (start_line, end_line) to nodes.
        """
        mapping: dict[tuple[int, int], ast.AST] = {}
        for node in ast.walk(tree):
            if not isinstance(node, kinds):
                continue
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", None)
            if lineno is None:
                continue
            start = int(lineno)
            end = int(end_lineno) if end_lineno is not None else start
            mapping[start, end] = node
        return cls(node_map=mapping)

    def lookup(self, start_line: int, end_line: int | None) -> ast.AST | None:
        """
        Return the node that spans the given lines, or the first enclosing span.

        Returns
        -------
        ast.AST | None
            Node matching/enclosing the span when present; otherwise None.
        """
        end = int(end_line) if end_line is not None else int(start_line)
        node = self.node_map.get((start_line, end))
        if node is not None:
            return node

        # Prefer nodes that enclose the requested start line.
        enclosing: ast.AST | None = None
        smallest_enclosing_span: tuple[int, int] | None = None

        # Also allow nodes that begin after the requested start (e.g., decorator spans widen start_line)
        # as long as the requested end is within the candidate span.
        overlap: ast.AST | None = None
        smallest_overlap_span: tuple[int, int] | None = None

        for (span_start, span_end), candidate in self.node_map.items():
            if span_start <= start_line <= span_end:
                if smallest_enclosing_span is None or (span_end - span_start) < (
                    smallest_enclosing_span[1] - smallest_enclosing_span[0]
                ):
                    smallest_enclosing_span = (span_start, span_end)
                    enclosing = candidate
                continue
            if start_line <= span_start <= end <= span_end and (
                smallest_overlap_span is None
                or (span_end - span_start) < (smallest_overlap_span[1] - smallest_overlap_span[0])
            ):
                smallest_overlap_span = (span_start, span_end)
                overlap = candidate

        if enclosing is not None:
            return enclosing
        if overlap is not None:
            return overlap
        return None


def parse_python_module(path: Path) -> tuple[list[str], ast.AST] | None:
    """
    Parse a Python module into an AST, returning source lines and the tree.

    Returns
    -------
    tuple[list[str], ast.AST] | None
        Lines and parsed AST when successful; None when the file is missing or invalid.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    return source.splitlines(), tree


def timed_parse(path: Path) -> tuple[list[str], ast.AST, float] | None:
    """
    Parse a Python file and return lines, AST, and duration seconds.

    Returns
    -------
    tuple[list[str], ast.AST, float] | None
        Lines, AST, and parse duration; None on failure.
    """
    start = time.perf_counter()
    parsed = parse_python_module(path)
    if parsed is None:
        return None
    lines, tree = parsed
    return lines, tree, time.perf_counter() - start
