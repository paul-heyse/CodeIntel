"""Resolve GOID spans to source slices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from codeintel.analytics.parsing.models import ParsedFunction, SourceSpan


@dataclass(frozen=True)
class SpanResolutionResult:
    """Resolved span details for a function GOID."""

    function_goid_h128: int
    span: SourceSpan
    path: Path


class SpanResolutionError(Exception):
    """Raised when a function GOID cannot be resolved to a span."""


def build_span_index(parsed_functions: Iterable[ParsedFunction]) -> Mapping[int, SourceSpan]:
    """
    Build a mapping from function_goid_h128 to SourceSpan.

    Functions without GOIDs are skipped.

    Returns
    -------
    Mapping[int, SourceSpan]
        Span index keyed by GOID hashes.
    """
    index: dict[int, SourceSpan] = {}
    for parsed in parsed_functions:
        if parsed.function_goid_h128 is None:
            continue
        index[parsed.function_goid_h128] = parsed.span
    return index


def resolve_span(
    *,
    function_goid_h128: int,
    span_index: Mapping[int, SourceSpan],
) -> SpanResolutionResult:
    """
    Resolve a function GOID to a SourceSpan.

    Parameters
    ----------
    function_goid_h128 :
        GOID hash for the target function.
    span_index :
        Mapping from GOID hash to source spans.

    Returns
    -------
    SpanResolutionResult
        Resolved span information including the source path.

    Raises
    ------
    SpanResolutionError
        If the GOID is not present in the span index.
    """
    try:
        span = span_index[function_goid_h128]
    except KeyError as exc:
        message = f"No span for function_goid_h128={function_goid_h128}"
        raise SpanResolutionError(message) from exc

    return SpanResolutionResult(
        function_goid_h128=function_goid_h128,
        span=span,
        path=span.path,
    )
