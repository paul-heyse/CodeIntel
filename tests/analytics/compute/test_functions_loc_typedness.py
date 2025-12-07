"""Tests for LOC and typedness computation."""

from __future__ import annotations

import ast

from codeintel.analytics.compute.functions.loc import LinesOfCode, compute_loc, count_logical_lines
from codeintel.analytics.compute.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)


def _parse_function(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_compute_loc_counts_lines() -> None:
    """Count physical, logical, blank, and comment lines."""
    lines = [
        "def func():",
        "    # comment line",
        "    value = 1",
        "    ",
        "    return value",
    ]
    loc = compute_loc(lines, start_line=1, end_line=len(lines))
    expected = LinesOfCode(physical=5, logical=3, blank=1, comment=1)
    assert loc == expected
    assert count_logical_lines(lines) == 3


def test_compute_param_stats_captures_annotations() -> None:
    """Extract parameter and return annotations with counts."""
    source = """
def demo(self, a: int, b, *, flag: bool = False, **kwargs) -> str:
    return str(a + (b or 0))
"""
    node = _parse_function(source)
    stats = compute_param_stats(node)
    expected_types = {"a": "int", "b": None, "flag": "bool", "kwargs": None}
    assert stats.param_count == 5
    assert stats.positional_params == 3
    assert stats.keyword_only_params == 1
    assert stats.has_varargs is False
    assert stats.has_varkw is True
    assert stats.total_params == 4
    assert stats.annotated_params == 2
    assert stats.param_types == expected_types
    assert stats.has_return_annotation is True
    assert stats.return_type == "str"


def test_compute_param_stats_non_function_defaults() -> None:
    """Non-function nodes return zeroed statistics."""
    stats = compute_param_stats(ast.parse("x = 1").body[0])
    assert stats == ParamStats(
        param_count=0,
        positional_params=0,
        keyword_only_params=0,
        has_varargs=False,
        has_varkw=False,
        total_params=0,
        annotated_params=0,
        param_types={},
        has_return_annotation=False,
        return_type=None,
    )


def test_compute_typedness_flags_variants() -> None:
    """Compute typedness flags for fully typed, partial, and untyped cases."""
    fully_typed = compute_typedness_flags(
        total_params=2, annotated_params=2, has_return_annotation=True
    )
    assert fully_typed == TypednessFlags(
        param_typed_ratio=1.0,
        unannotated_params=0,
        fully_typed=True,
        partial_typed=False,
        untyped=False,
        typedness_bucket="typed",
        typedness_source="annotations",
    )

    partial = compute_typedness_flags(
        total_params=3, annotated_params=1, has_return_annotation=False
    )
    assert partial.partial_typed is True
    assert partial.typedness_bucket == "partial"
    assert partial.unannotated_params == 2

    untyped = compute_typedness_flags(
        total_params=1, annotated_params=0, has_return_annotation=False
    )
    assert untyped.untyped is True
    assert untyped.typedness_bucket == "untyped"
