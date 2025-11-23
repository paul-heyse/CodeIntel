"""Tests for FunctionSpanIndex span resolution."""

from __future__ import annotations

from codeintel.graphs.function_index import FunctionSpan, FunctionSpanIndex

EXACT_GOID_FOO = 1
EXACT_GOID_BAR = 2


def test_lookup_prefers_exact_span_and_qualname() -> None:
    """
    Exact spans and matching qualnames are preferred.

    Raises
    ------
    AssertionError
        If resolution returns the wrong GOID.
    """
    spans = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=EXACT_GOID_FOO,
                rel_path="pkg/a.py",
                qualname="pkg.a.foo",
                start_line=10,
                end_line=20,
            ),
            FunctionSpan(
                goid=EXACT_GOID_BAR,
                rel_path="pkg/a.py",
                qualname="pkg.a.bar",
                start_line=30,
                end_line=40,
            ),
        ]
    )

    result_foo = spans.lookup("pkg/a.py", 10, 20, "pkg.a.foo")
    if result_foo != EXACT_GOID_FOO:
        message = f"expected GOID {EXACT_GOID_FOO} for foo, got {result_foo}"
        raise AssertionError(message)

    result_bar = spans.lookup("pkg/a.py", 30, 40, "pkg.a.bar")
    if result_bar != EXACT_GOID_BAR:
        message = f"expected GOID {EXACT_GOID_BAR} for bar, got {result_bar}"
        raise AssertionError(message)


def test_lookup_falls_back_to_enclosing_and_same_line() -> None:
    """
    Lookup should fall back through enclosure and line-based heuristics.

    Raises
    ------
    AssertionError
        If fallback ordering changes unexpectedly.
    """
    outer_goid = 1
    inner_goid = 2
    spans = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=outer_goid,
                rel_path="pkg/a.py",
                qualname="pkg.a.outer",
                start_line=5,
                end_line=50,
            ),
            FunctionSpan(
                goid=inner_goid,
                rel_path="pkg/a.py",
                qualname="pkg.a.inner",
                start_line=10,
                end_line=15,
            ),
        ]
    )

    match_inner = spans.lookup("pkg/a.py", 12, 12, "pkg.a.inner")
    if match_inner != inner_goid:
        message = f"expected inner GOID {inner_goid}, got {match_inner}"
        raise AssertionError(message)

    fallback_outer = spans.lookup("pkg/a.py", 12, 12, "pkg.a.unknown")
    if fallback_outer != outer_goid:
        message = f"expected outer GOID {outer_goid}, got {fallback_outer}"
        raise AssertionError(message)

    start_only = spans.lookup("pkg/a.py", 5, None, "pkg.a.outer")
    if start_only != outer_goid:
        message = f"expected outer GOID {outer_goid} for start-only lookup, got {start_only}"
        raise AssertionError(message)

    no_match = spans.lookup("pkg/b.py", 1, 1, None)
    if no_match is not None:
        message = f"expected None for unrelated path, got {no_match}"
        raise AssertionError(message)
