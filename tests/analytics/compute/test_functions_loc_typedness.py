"""Tests for LOC and typedness computation."""

from __future__ import annotations

import ast
from typing import cast

from codeintel.analytics.compute.functions.loc import LinesOfCode, compute_loc, count_logical_lines
from codeintel.analytics.compute.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)


def _parse_function(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = tree.body[0]
    expect_is_instance(node, ast.FunctionDef, label="parsed node type")
    return cast("ast.FunctionDef", node)


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
    expect_equal(loc, expected, label="loc counts")
    expect_equal(count_logical_lines(lines), 3, label="logical lines")


def test_compute_param_stats_captures_annotations() -> None:
    """Extract parameter and return annotations with counts."""
    source = """
def demo(self, a: int, b, *, flag: bool = False, **kwargs) -> str:
    return str(a + (b or 0))
    """
    node = _parse_function(source)
    stats = compute_param_stats(node)
    expected_types = {"a": "int", "b": None, "flag": "bool", "kwargs": None}
    expect_equal(stats.param_count, 5, label="param_count")
    expect_equal(stats.positional_params, 3, label="positional_params")
    expect_equal(stats.keyword_only_params, 1, label="keyword_only_params")
    expect_true(stats.has_varargs is False, message="has_varargs")
    expect_true(stats.has_varkw is True, message="has_varkw")
    expect_equal(stats.total_params, 4, label="total_params")
    expect_equal(stats.annotated_params, 2, label="annotated_params")
    expect_equal(stats.param_types, expected_types, label="param_types")
    expect_true(stats.has_return_annotation is True, message="return annotation")
    expect_equal(stats.return_type, "str", label="return_type")


def test_compute_param_stats_non_function_defaults() -> None:
    """Non-function nodes return zeroed statistics."""
    stats = compute_param_stats(ast.parse("x = 1").body[0])
    expect_equal(
        stats,
        ParamStats(
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
        ),
    )


def test_compute_typedness_flags_variants() -> None:
    """Compute typedness flags for fully typed, partial, and untyped cases."""
    fully_typed = compute_typedness_flags(
        total_params=2, annotated_params=2, has_return_annotation=True
    )
    expect_equal(
        fully_typed,
        TypednessFlags(
            param_typed_ratio=1.0,
            unannotated_params=0,
            fully_typed=True,
            partial_typed=False,
            untyped=False,
            typedness_bucket="typed",
            typedness_source="annotations",
        ),
    )

    partial = compute_typedness_flags(
        total_params=3, annotated_params=1, has_return_annotation=False
    )
    expect_true(partial.partial_typed is True, message="partial_typed")
    expect_equal(partial.typedness_bucket, "partial", label="typedness_bucket")
    expect_equal(partial.unannotated_params, 2, label="unannotated_params")

    untyped = compute_typedness_flags(
        total_params=1, annotated_params=0, has_return_annotation=False
    )
    expect_true(untyped.untyped is True, message="untyped flag")
    expect_equal(untyped.typedness_bucket, "untyped", label="typedness_bucket")
