"""AST span index for fast node lookup.

This module provides the canonical AstSpanIndex class for indexing
AST nodes by line spans.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class AstSpanIndex:
    r"""Index AST nodes by (start_line, end_line) spans for quick lookup.

    This index provides O(1) exact span lookup and O(n) enclosing span
    lookup for efficient AST traversal.

    Attributes
    ----------
    node_map
        Mapping from (start_line, end_line) tuples to AST nodes.

    Examples
    --------
    >>> import ast
    >>> tree = ast.parse("def foo():\n    pass")
    >>> index = AstSpanIndex.from_tree(tree, (ast.FunctionDef, ast.AsyncFunctionDef))
    >>> index.lookup(1, 2)  # doctest: +ELLIPSIS
    <ast.FunctionDef object at ...>
    """

    node_map: dict[tuple[int, int], ast.AST]

    @classmethod
    def from_tree(
        cls,
        tree: ast.AST,
        kinds: tuple[type[ast.AST], ...],
    ) -> AstSpanIndex:
        """Build an index for the given AST and target node kinds.

        Parameters
        ----------
        tree
            Root AST node to index.
        kinds
            Tuple of AST node types to include in the index.

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

    def lookup(self, start_line: int, end_line: int | None = None) -> ast.AST | None:
        """Return the node that spans the given lines, or the first enclosing span.

        Resolution order:
        1. Exact span match (start_line, end_line)
        2. Smallest enclosing span
        3. Smallest overlapping span

        Parameters
        ----------
        start_line
            Starting line number to search for.
        end_line
            Optional ending line number. Defaults to start_line.

        Returns
        -------
        ast.AST | None
            Node matching/enclosing the span when present; otherwise None.
        """
        end = int(end_line) if end_line is not None else int(start_line)

        # Check for exact match
        node = self.node_map.get((start_line, end))
        if node is not None:
            return node

        # Search for enclosing or overlapping spans
        enclosing: ast.AST | None = None
        smallest_enclosing_span: tuple[int, int] | None = None

        overlap: ast.AST | None = None
        smallest_overlap_span: tuple[int, int] | None = None

        for (span_start, span_end), candidate in self.node_map.items():
            # Check for enclosing span
            if span_start <= start_line <= span_end:
                if smallest_enclosing_span is None or (span_end - span_start) < (
                    smallest_enclosing_span[1] - smallest_enclosing_span[0]
                ):
                    smallest_enclosing_span = (span_start, span_end)
                    enclosing = candidate
                continue

            # Check for overlapping span
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

    def __len__(self) -> int:
        """Return the number of indexed nodes.

        Returns
        -------
        int
            Total number of nodes in the index.
        """
        return len(self.node_map)


__all__ = [
    "AstSpanIndex",
]
