"""IntervalTree-backed span resolution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, cast

from intervaltree import IntervalTree

from codeintel.core.paths import normalize_path
from codeintel.core.spans import normalize_byte_span, normalize_line_span, to_half_open_span

MatchKind = Literal["EXACT", "POINT", "POINT_ADJACENT", "CONTAINS", "OVERLAP", "NONE"]


@dataclass(frozen=True)
class SpanMatch[T]:
    """Result of resolving a span to a payload."""

    payload: T | None
    match_kind: MatchKind
    candidate_count: int


@dataclass(frozen=True)
class _SpanCandidate[T]:
    start: int
    end: int
    payload: T
    order: int


@dataclass
class _PathIndex[T]:
    tree: IntervalTree = field(default_factory=IntervalTree)
    exact: dict[tuple[int, int], list[_SpanCandidate[T]]] = field(default_factory=dict)


def _line_span(start: int | None, end: int | None) -> tuple[int, int] | None:
    if start is None:
        return None
    start_value, end_value = normalize_line_span(start, end)
    return to_half_open_span(start_value, end_value)


def _byte_span(start: int | None, end: int | None) -> tuple[int, int] | None:
    return normalize_byte_span(start, end)


class SpanResolver[T]:
    """Resolve spans to payloads using IntervalTree."""

    def __init__(
        self,
        *,
        span_normalizer: Callable[[int | None, int | None], tuple[int, int] | None],
        path_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._normalize_span = span_normalizer
        self._normalize_path = path_normalizer or normalize_path
        self._index: dict[str, _PathIndex[T]] = {}
        self._order = 0

    @classmethod
    def for_lines(cls, *, path_normalizer: Callable[[str], str] | None = None) -> SpanResolver[T]:
        """Create a resolver for line-based spans.

        Returns
        -------
        SpanResolver[T]
            Resolver configured for line spans.
        """
        return cls(span_normalizer=_line_span, path_normalizer=path_normalizer)

    @classmethod
    def for_bytes(cls, *, path_normalizer: Callable[[str], str] | None = None) -> SpanResolver[T]:
        """Create a resolver for byte-based spans.

        Returns
        -------
        SpanResolver[T]
            Resolver configured for byte spans.
        """
        return cls(span_normalizer=_byte_span, path_normalizer=path_normalizer)

    def add_span(self, path: str, start: int | None, end: int | None, payload: T) -> None:
        """Add a single span payload to the resolver."""
        normalized = self._normalize_span(start, end)
        if normalized is None:
            return
        start_value, end_value = normalized
        path_key = self._normalize_path(path)
        index = self._index.setdefault(path_key, _PathIndex())
        candidate = _SpanCandidate(
            start=start_value,
            end=end_value,
            payload=payload,
            order=self._order,
        )
        self._order += 1
        index.exact.setdefault((start_value, end_value), []).append(candidate)
        if end_value > start_value:
            index.tree.addi(start_value, end_value, candidate)

    def add_spans(self, spans: Iterable[tuple[str, int | None, int | None, T]]) -> None:
        """Add multiple span payloads to the resolver."""
        for path, start, end, payload in spans:
            self.add_span(path, start, end, payload)

    def resolve_candidates(
        self,
        path: str,
        start: int | None,
        end: int | None,
        *,
        allow_adjacent_point: bool = False,
    ) -> tuple[list[T], MatchKind]:
        """Resolve a span to candidate payloads and match kind.

        Returns
        -------
        tuple[list[T], MatchKind]
            Ordered payloads and the match kind for the span.
        """
        normalized = self._normalize_span(start, end)
        if normalized is None:
            return [], "NONE"
        path_key = self._normalize_path(path)
        index = self._index.get(path_key)
        if index is None:
            return [], "NONE"
        start_value, end_value = normalized
        match_kind: MatchKind = "NONE"
        candidates: list[T] = []
        exact = index.exact.get((start_value, end_value))
        if exact:
            candidates = _payloads_from_candidates(exact)
            match_kind = "EXACT"
        elif start_value == end_value:
            candidates = cast("list[T]", _interval_payloads(index.tree, start_value, start_value))
            if candidates:
                match_kind = "POINT"
            elif allow_adjacent_point and start_value > 0:
                candidates = cast(
                    "list[T]",
                    _interval_payloads(index.tree, start_value - 1, start_value - 1),
                )
                if candidates:
                    match_kind = "POINT_ADJACENT"
        else:
            candidates = cast("list[T]", _enveloped_payloads(index.tree, start_value, end_value))
            if candidates:
                match_kind = "CONTAINS"
            else:
                candidates = cast("list[T]", _overlap_payloads(index.tree, start_value, end_value))
                if candidates:
                    match_kind = "OVERLAP"
        return candidates, match_kind

    def resolve(
        self,
        path: str,
        start: int | None,
        end: int | None,
        *,
        allow_adjacent_point: bool = False,
    ) -> SpanMatch[T]:
        """Resolve a span to a single payload and match metadata.

        Returns
        -------
        SpanMatch[T]
            Match payload, kind, and candidate count.
        """
        candidates, match_kind = self.resolve_candidates(
            path,
            start,
            end,
            allow_adjacent_point=allow_adjacent_point,
        )
        payload = candidates[0] if candidates else None
        return SpanMatch(payload=payload, match_kind=match_kind, candidate_count=len(candidates))


def _payloads_from_candidates[T](candidates: Iterable[_SpanCandidate[T]]) -> list[T]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.end - candidate.start, candidate.order),
    )
    return [candidate.payload for candidate in ordered]


def _interval_candidates(intervals: Iterable[object]) -> list[_SpanCandidate[object]]:
    candidates: list[_SpanCandidate[object]] = []
    for interval in intervals:
        data = getattr(interval, "data", None)
        if isinstance(data, _SpanCandidate):
            candidates.append(data)
    return candidates


def _interval_payloads(
    tree: IntervalTree,
    start: int,
    end: int,
) -> list[object]:
    overlaps = tree.overlap(start, end + 1)
    return _payloads_from_candidates(_interval_candidates(overlaps))


def _enveloped_payloads(
    tree: IntervalTree,
    start: int,
    end: int,
) -> list[object]:
    envelop = getattr(tree, "envelop", None)
    if callable(envelop):
        intervals = envelop(start, end)
        if isinstance(intervals, Iterable):
            return _payloads_from_candidates(_interval_candidates(intervals))
        return []
    overlaps = tree.overlap(start, end)
    contained: list[object] = []
    for interval in overlaps:
        begin = getattr(interval, "begin", None)
        finish = getattr(interval, "end", None)
        if isinstance(begin, int) and isinstance(finish, int) and begin <= start and finish >= end:
            contained.append(interval)
    return _payloads_from_candidates(_interval_candidates(contained))


def _overlap_payloads(
    tree: IntervalTree,
    start: int,
    end: int,
) -> list[object]:
    return _payloads_from_candidates(_interval_candidates(tree.overlap(start, end)))
