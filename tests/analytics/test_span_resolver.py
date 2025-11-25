"""Tests for span resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.parsing.models import ParsedFunction, SourceSpan
from codeintel.analytics.parsing.span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)


def test_build_span_index_skips_missing_goids(tmp_path: Path) -> None:
    """Only functions with GOIDs are included in the span index."""
    path = tmp_path / "mod.py"
    goid = 123
    span = SourceSpan(path=path, start_line=1, start_col=0, end_line=2, end_col=1)
    parsed = ParsedFunction(
        path=path,
        qualname="mod.foo",
        function_goid_h128=goid,
        span=span,
        ast=None,
        docstring=None,
        param_annotations={},
        return_annotation=None,
        param_any_flags={},
        return_is_any=False,
    )
    orphan = ParsedFunction(
        path=path,
        qualname="mod.bar",
        function_goid_h128=None,
        span=span,
        ast=None,
        docstring=None,
        param_annotations={},
        return_annotation=None,
        param_any_flags={},
        return_is_any=False,
    )

    index = build_span_index([parsed, orphan])

    if goid not in index:
        pytest.fail("Expected GOID to be present in span index")
    if len(index) != 1:
        pytest.fail(f"Expected only one indexed span, got {len(index)}")
    if index[goid] != span:
        pytest.fail("Indexed span did not match expected span")


def test_resolve_span_success(tmp_path: Path) -> None:
    """Resolve a span successfully when present in the index."""
    path = tmp_path / "mod.py"
    span = SourceSpan(path=path, start_line=10, start_col=0, end_line=12, end_col=1)
    result = resolve_span(function_goid_h128=1, span_index={1: span})

    if not isinstance(result, SpanResolutionResult):
        pytest.fail("Expected resolve_span to return a SpanResolutionResult")
    if result.span != span:
        pytest.fail("Resolved span mismatch")
    if result.path != path:
        pytest.fail("Resolved path mismatch")


def test_resolve_span_missing() -> None:
    """Raise SpanResolutionError when span is absent."""
    with pytest.raises(SpanResolutionError):
        resolve_span(function_goid_h128=99, span_index={})
