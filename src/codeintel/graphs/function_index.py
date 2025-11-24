"""Shared helpers for mapping source spans to GOIDs."""

from __future__ import annotations

from collections.abc import Callable as TypingCallable
from collections.abc import Iterable
from dataclasses import dataclass

from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path


@dataclass(frozen=True)
class FunctionSpan:
    """Normalized function metadata used by graph builders."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


class FunctionSpanIndex:
    """Lookup structure for resolving GOIDs from file spans."""

    def __init__(self, spans: Iterable[FunctionSpan]) -> None:
        self._by_path: dict[str, list[FunctionSpan]] = {}
        for span in spans:
            path = normalize_rel_path(span.rel_path)
            self._by_path.setdefault(path, []).append(span)

        for path_spans in self._by_path.values():
            path_spans.sort(key=lambda s: (s.start_line, s.end_line))

    def paths(self) -> list[str]:
        """
        Paths with at least one function span.

        Returns
        -------
        list[str]
            Paths present in the index.
        """
        return list(self._by_path.keys())

    def spans_for_path(self, rel_path: str) -> list[FunctionSpan]:
        """
        Return spans for a given relative path.

        Returns
        -------
        list[FunctionSpan]
            Spans for the requested path (empty when missing).
        """
        return list(self._by_path.get(normalize_rel_path(rel_path), []))

    def local_name_map(self, rel_path: str) -> dict[str, int]:
        """
        Map local names and qualnames to GOIDs for a single file.

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

    def lookup(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a GOID for the given path and span.

        Resolution order favors exact span matches, then qualname matches
        overlapping the span, then any enclosing span, and finally a fallback
        to functions starting on the same line.

        Returns
        -------
        int | None
            GOID when found; otherwise None.
        """
        spans_list = self._by_path.get(normalize_rel_path(rel_path))
        if spans_list is None:
            return None
        spans: list[FunctionSpan] = spans_list
        if not spans:
            return None

        start = int(start_line)
        end = int(end_line) if end_line is not None else start

        def _first_match(predicate: TypingCallable[[FunctionSpan], bool]) -> int | None:
            for span in spans:
                if predicate(span):
                    return span.goid
            return None

        predicates: list[TypingCallable[[FunctionSpan], bool]] = []
        if qualname:
            predicates.append(
                lambda span: span.start_line == start
                and span.end_line == end
                and _qualname_matches(span.qualname, qualname)
            )
        predicates.append(lambda span: span.start_line == start and span.end_line == end)
        if qualname:
            predicates.append(
                lambda span: _qualname_matches(span.qualname, qualname)
                and span.start_line <= start <= span.end_line
            )
        predicates.append(lambda span: span.start_line <= start <= span.end_line)
        predicates.append(lambda span: span.start_line == start)

        for predicate in predicates:
            match = _first_match(predicate)
            if match is not None:
                return match
        return None


def _qualname_matches(full: str, candidate: str) -> bool:
    if full == candidate:
        return True
    suffix = candidate.rsplit(".", maxsplit=1)[-1]
    return full.endswith(f".{suffix}")


def load_function_spans(gateway: StorageGateway, *, repo: str, commit: str) -> list[FunctionSpan]:
    """
    Load function spans from `core.goids` for a repo snapshot.

    Returns
    -------
    list[FunctionSpan]
        Normalized function spans keyed by GOID.
    """
    con = gateway.con
    rows = con.execute(
        """
        SELECT goid_h128, rel_path, qualname, start_line, end_line
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetchall()

    spans: list[FunctionSpan] = []
    for goid_h128, rel_path, qualname, start_line, end_line in rows:
        if start_line is None:
            continue
        spans.append(
            FunctionSpan(
                goid=int(goid_h128),
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else int(start_line),
            )
        )
    return spans


def load_function_index(gateway: StorageGateway, *, repo: str, commit: str) -> FunctionSpanIndex:
    """
    Create a `FunctionSpanIndex` from DuckDB state.

    Returns
    -------
    FunctionSpanIndex
        Index seeded from `core.goids` for the repo/commit snapshot.
    """
    return FunctionSpanIndex(load_function_spans(gateway, repo=repo, commit=commit))
