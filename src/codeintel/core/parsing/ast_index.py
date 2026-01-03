"""AST span index for fast node lookup.

This module provides the canonical AstSpanIndex class for indexing
AST nodes by line spans.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field

from intervaltree import IntervalTree

from codeintel.core.spans import to_half_open_span

_SPAN_TUPLE_LEN = 3


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
    _tree: IntervalTree = field(default_factory=IntervalTree, repr=False)
    _order: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)

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
        interval_tree = IntervalTree()
        order_map: dict[tuple[int, int], int] = {}
        order = 0
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
            order_map.setdefault((start, end), order)
            order += 1
            span_start, span_end = to_half_open_span(start, end)
            if span_end > span_start:
                interval_tree.addi(span_start, span_end, (start, end, node))
        return cls(node_map=mapping, _tree=interval_tree, _order=order_map)

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

        query_start, query_end = to_half_open_span(start_line, end)
        enclosing = self._best_candidate(self._enclosing_intervals(query_start, query_end))
        if enclosing is not None:
            return enclosing
        overlap = self._best_candidate(self._tree.overlap(query_start, query_end))
        if overlap is not None:
            return overlap
        return None

    def _enclosing_intervals(self, start: int, end: int) -> list[object]:
        envelop = getattr(self._tree, "envelop", None)
        if callable(envelop):
            intervals = envelop(start, end)
            if isinstance(intervals, Iterable):
                return list(intervals)
            return []
        contained: list[object] = []
        for interval in self._tree.overlap(start, end):
            begin = getattr(interval, "begin", None)
            finish = getattr(interval, "end", None)
            if (
                isinstance(begin, int)
                and isinstance(finish, int)
                and begin <= start
                and finish >= end
            ):
                contained.append(interval)
        return contained

    def _best_candidate(self, intervals: Iterable[object]) -> ast.AST | None:
        candidates: list[tuple[int, int, int, ast.AST]] = []
        for interval in intervals:
            data = getattr(interval, "data", None)
            if not isinstance(data, tuple) or len(data) != _SPAN_TUPLE_LEN:
                continue
            span_start, span_end, node = data
            if not isinstance(node, ast.AST):
                continue
            key = (span_start, span_end)
            order = self._order.get(key, 0)
            candidates.append((span_end - span_start, order, span_start, node))
        if not candidates:
            return None
        _, _, _, best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return best

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
