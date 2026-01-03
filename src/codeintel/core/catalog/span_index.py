"""Span index for efficient function lookup.

This module provides the SpanIndex class for fast GOID resolution
from file paths and line numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from intervaltree import IntervalTree

from codeintel.core.paths import normalize_path
from codeintel.core.spans import to_half_open_span

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from codeintel.core.catalog.function_span import FunctionSpan


def _qualname_matches(full: str, candidate: str) -> bool:
    """Check if a qualname matches a candidate.

    Parameters
    ----------
    full
        Full qualified name from the span.
    candidate
        Candidate name to match against.

    Returns
    -------
    bool
        True if the candidate matches the full qualname.
    """
    if full == candidate:
        return True
    suffix = candidate.rsplit(".", maxsplit=1)[-1]
    return full.endswith(f".{suffix}")


class SpanIndex:
    """Lookup structure for resolving GOIDs from file spans.

    This index provides fast lookups of function GOIDs based on file
    path and line numbers, with support for qualname disambiguation.

    Examples
    --------
    >>> spans = [FunctionSpan(goid=1, rel_path="a.py", qualname="foo", start_line=1, end_line=5)]
    >>> index = SpanIndex(spans)
    >>> index.lookup("a.py", 3)
    1
    """

    def __init__(
        self,
        spans: Iterable[FunctionSpan],
        *,
        path_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the index from an iterable of function spans.

        Parameters
        ----------
        spans
            Function spans to index.
        path_normalizer
            Optional function to normalize paths. Defaults to normalize_path.
        """
        self._normalize = path_normalizer or normalize_path
        self._by_path: dict[str, list[FunctionSpan]] = {}
        self._tree_by_path: dict[str, IntervalTree] = {}

        for span in spans:
            path = self._normalize(span.rel_path)
            self._by_path.setdefault(path, []).append(span)

        # Sort spans by line for consistent lookup order
        for path, path_spans in self._by_path.items():
            path_spans.sort(key=lambda s: (s.start_line, s.end_line))
            tree = IntervalTree()
            for span in path_spans:
                start, end = to_half_open_span(span.start_line, span.end_line)
                tree.addi(start, end, span)
            self._tree_by_path[path] = tree

    @staticmethod
    def _sorted_spans(matches: Iterable[object]) -> list[FunctionSpan]:
        spans: list[FunctionSpan] = []
        for match in matches:
            span = getattr(match, "data", None)
            if span is not None:
                spans.append(span)
        spans.sort(key=lambda s: (s.end_line - s.start_line, s.start_line, s.end_line))
        return spans

    def paths(self) -> list[str]:
        """Return paths with at least one function span.

        Returns
        -------
        list[str]
            Paths present in the index.
        """
        return list(self._by_path.keys())

    def spans_for_path(self, rel_path: str) -> list[FunctionSpan]:
        """Return spans for a given relative path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        list[FunctionSpan]
            Spans for the requested path (empty when missing).
        """
        return list(self._by_path.get(self._normalize(rel_path), []))

    def local_name_map(self, rel_path: str) -> dict[str, int]:
        """Map local names and qualnames to GOIDs for a single file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        dict[str, int]
            Mapping from short/qualified names to GOIDs.
        """
        mapping: dict[str, int] = {}
        for span in self.spans_for_path(rel_path):
            local_name = span.qualname.rsplit(".", maxsplit=1)[-1]
            mapping.setdefault(local_name, span.goid)
            mapping.setdefault(span.qualname, span.goid)
        return mapping

    @staticmethod
    def _lookup_predicates(
        *,
        start: int,
        end: int,
        qualname: str | None,
    ) -> list[Callable[[FunctionSpan], bool]]:
        """Return predicate functions for span lookup.

        Returns
        -------
        list[Callable[[FunctionSpan], bool]]
            Predicate functions in priority order.
        """
        predicates: list[Callable[[FunctionSpan], bool]] = []

        if qualname:
            exact_qualname = qualname

            def _exact_span_with_qualname(span: FunctionSpan) -> bool:
                return (
                    span.start_line == start
                    and span.end_line == end
                    and _qualname_matches(span.qualname, exact_qualname)
                )

            predicates.append(_exact_span_with_qualname)

        def _exact_span(span: FunctionSpan) -> bool:
            return span.start_line == start and span.end_line == end

        predicates.append(_exact_span)

        if qualname:
            overlap_qualname = qualname

            def _overlap_qualname_match(span: FunctionSpan) -> bool:
                return _qualname_matches(span.qualname, overlap_qualname) and (
                    span.start_line <= start <= span.end_line
                )

            predicates.append(_overlap_qualname_match)

        def _enclosing_span(span: FunctionSpan) -> bool:
            return span.start_line <= start <= span.end_line

        predicates.append(_enclosing_span)

        def _start_line_only(span: FunctionSpan) -> bool:
            return span.start_line == start

        predicates.append(_start_line_only)

        return predicates

    def lookup(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """Resolve a GOID for the given path and span.

        Resolution order favors exact span matches, then qualname matches
        overlapping the span, then any enclosing span, and finally a fallback
        to functions starting on the same line.

        Parameters
        ----------
        rel_path
            Relative file path.
        start_line
            Starting line number.
        end_line
            Optional ending line number. Defaults to start_line.
        qualname
            Optional qualified name for disambiguation.

        Returns
        -------
        int | None
            GOID when found; otherwise None.
        """
        path = self._normalize(rel_path)
        tree = self._tree_by_path.get(path)
        if tree is None:
            return None
        start = int(start_line)
        end = int(end_line) if end_line is not None else start
        query_start, query_end = to_half_open_span(start, end)
        candidates = self._sorted_spans(tree.overlap(query_start, query_end))
        if not candidates:
            return None

        def _first_match(predicate: Callable[[FunctionSpan], bool]) -> int | None:
            for span in candidates:
                if predicate(span):
                    return span.goid
            return None

        for predicate in self._lookup_predicates(start=start, end=end, qualname=qualname):
            match = _first_match(predicate)
            if match is not None:
                return match
        return None

    def __len__(self) -> int:
        """Return total number of spans in the index.

        Returns
        -------
        int
            Total number of function spans across all files.
        """
        return sum(len(spans) for spans in self._by_path.values())


__all__ = [
    "SpanIndex",
    "normalize_path",
]
