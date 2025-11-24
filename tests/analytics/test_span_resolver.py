"""Tests for span resolution helper."""

from __future__ import annotations

import ast

import pytest

from codeintel.analytics.span_resolver import SpanResolution, resolve_span
from codeintel.ingestion.ast_utils import AstSpanIndex


def _index_for_source(source: str) -> AstSpanIndex:
    tree = ast.parse(source)
    return AstSpanIndex.from_tree(tree, (ast.FunctionDef, ast.AsyncFunctionDef))


def test_resolve_span_ok() -> None:
    """Resolve a span successfully when the index contains the function."""
    index = _index_for_source("""\
def foo():
    return 1
""")
    resolution = resolve_span(index, 1, 2)
    if not isinstance(resolution, SpanResolution):
        pytest.fail("Resolution should return a SpanResolution instance")
    if resolution.reason != "ok":
        pytest.fail(f"Expected reason 'ok', got '{resolution.reason}'")
    if resolution.node is None:
        pytest.fail("Expected a resolved AST node for the provided span")
    if resolution.detail is not None:
        pytest.fail(f"Expected no detail on success, got: {resolution.detail}")


def test_resolve_span_missing_span() -> None:
    """Return a missing-span result when the range is absent."""
    index = _index_for_source("""\
def foo():
    return 1
""")
    resolution = resolve_span(index, 10, 12)
    if resolution.node is not None:
        pytest.fail("Expected no node when span is missing")
    if resolution.reason != "missing_span":
        pytest.fail(f"Expected reason 'missing_span', got '{resolution.reason}'")
    if resolution.detail != "Span 10-12":
        pytest.fail(f"Unexpected detail payload: {resolution.detail}")


def test_resolve_span_missing_index() -> None:
    """Return a missing-index result when no span index is provided."""
    resolution = resolve_span(None, 1, 1)
    if resolution.node is not None:
        pytest.fail("Expected no node when index is missing")
    if resolution.reason != "missing_index":
        pytest.fail(f"Expected reason 'missing_index', got '{resolution.reason}'")
    if resolution.detail != "No span index available":
        pytest.fail(f"Unexpected detail payload: {resolution.detail}")
