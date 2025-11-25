"""Compatibility shim re-exporting span resolution helpers."""

from __future__ import annotations

from codeintel.analytics.parsing.span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)

__all__ = [
    "SpanResolutionError",
    "SpanResolutionResult",
    "build_span_index",
    "resolve_span",
]
