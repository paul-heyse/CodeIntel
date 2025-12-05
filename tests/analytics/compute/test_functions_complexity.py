"""Tests for complexity computation.

This module tests:
- ComplexityMetrics dataclass
- compute_complexity function
- Various AST node complexity counts
"""

from __future__ import annotations

import ast

from codeintel.analytics.compute.functions.complexity import (
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    ComplexityMetrics,
    compute_complexity,
)
from tests._helpers import assert_frozen

# Test constants
SIMPLE_FUNC_COMPLEXITY = 1
IF_ELSE_COMPLEXITY = 2
NESTED_IF_COMPLEXITY = 3


def _parse_function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse source and return the function node.

    Parameters
    ----------
    source
        Python source code containing a function definition.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef
        The parsed function node.
    """
    tree = ast.parse(source)
    return tree.body[0]  # type: ignore[return-value]


class TestComplexityMetrics:
    """Tests for ComplexityMetrics dataclass."""

    @staticmethod
    def test_creates_metrics() -> None:
        """Verify ComplexityMetrics can be created."""
        metrics = ComplexityMetrics(
            cyclomatic=1,
            max_nesting_depth=0,
            return_count=1,
            yield_count=0,
            raise_count=0,
            stmt_count=1,
            decorator_count=0,
            has_docstring=False,
            is_async=False,
            is_generator=False,
            complexity_bucket="low",
        )
        assert metrics.cyclomatic == 1

    @staticmethod
    def test_metrics_is_frozen() -> None:
        """Verify ComplexityMetrics is immutable."""
        metrics = ComplexityMetrics(
            cyclomatic=1,
            max_nesting_depth=0,
            return_count=1,
            yield_count=0,
            raise_count=0,
            stmt_count=1,
            decorator_count=0,
            has_docstring=False,
            is_async=False,
            is_generator=False,
            complexity_bucket="low",
        )
        assert_frozen(metrics, "cyclomatic", 5)


class TestComputeComplexityBasic:
    """Tests for basic compute_complexity functionality."""

    @staticmethod
    def test_simple_function() -> None:
        """Verify simple function has complexity 1."""
        source = """
def simple():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic == SIMPLE_FUNC_COMPLEXITY

    @staticmethod
    def test_function_with_if() -> None:
        """Verify if statement adds complexity."""
        source = """
def with_if(x):
    if x > 0:
        return x
    return -x
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic == IF_ELSE_COMPLEXITY

    @staticmethod
    def test_function_with_nested_if() -> None:
        """Verify nested if adds complexity."""
        source = """
def nested_if(x, y):
    if x > 0:
        if y > 0:
            return x + y
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic == NESTED_IF_COMPLEXITY


class TestComputeComplexityControlFlow:
    """Tests for control flow complexity."""

    @staticmethod
    def test_for_loop_adds_complexity() -> None:
        """Verify for loop adds complexity."""
        source = """
def with_for(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic >= IF_ELSE_COMPLEXITY

    @staticmethod
    def test_while_loop_adds_complexity() -> None:
        """Verify while loop adds complexity."""
        source = """
def with_while(x):
    while x > 0:
        x -= 1
    return x
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic >= IF_ELSE_COMPLEXITY

    @staticmethod
    def test_try_except_adds_complexity() -> None:
        """Verify try-except adds complexity."""
        source = """
def with_try():
    try:
        return 1
    except Exception:
        return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.cyclomatic >= IF_ELSE_COMPLEXITY

    @staticmethod
    def test_and_or_add_complexity() -> None:
        """Verify and/or in conditions add complexity."""
        source = """
def with_and_or(x, y, z):
    if x and y or z:
        return 1
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        # and and or each add +1
        assert metrics.cyclomatic >= NESTED_IF_COMPLEXITY


class TestComputeComplexityNesting:
    """Tests for nesting depth computation."""

    @staticmethod
    def test_flat_function_depth_zero() -> None:
        """Verify flat function has nesting depth 0."""
        source = """
def flat():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.max_nesting_depth == 0

    @staticmethod
    def test_single_if_depth_one() -> None:
        """Verify single if has nesting depth 1."""
        source = """
def with_if(x):
    if x > 0:
        return x
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.max_nesting_depth == 1

    @staticmethod
    def test_nested_depth() -> None:
        """Verify nested structures increase depth."""
        source = """
def deeply_nested(x, y, z):
    if x:
        if y:
            if z:
                return 1
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        expected_depth = 3
        assert metrics.max_nesting_depth == expected_depth


class TestComputeComplexityCounts:
    """Tests for statement counts."""

    @staticmethod
    def test_return_count() -> None:
        """Verify return statements are counted."""
        source = """
def multi_return(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        expected_returns = 3
        assert metrics.return_count == expected_returns

    @staticmethod
    def test_yield_count() -> None:
        """Verify yield statements are counted."""
        source = """
def generator(items):
    for item in items:
        yield item
        yield item * 2
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        expected_yields = 2
        assert metrics.yield_count == expected_yields
        assert metrics.is_generator is True

    @staticmethod
    def test_raise_count() -> None:
        """Verify raise statements are counted."""
        source = """
def raiser(x):
    if x < 0:
        raise ValueError("negative")
    if x > 100:
        raise ValueError("too large")
    return x
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        expected_raises = 2
        assert metrics.raise_count == expected_raises


class TestComputeComplexityAttributes:
    """Tests for function attribute detection."""

    @staticmethod
    def test_has_docstring() -> None:
        """Verify docstring detection."""
        source = '''
def documented():
    """This function has a docstring."""
    return 1
'''
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.has_docstring is True

    @staticmethod
    def test_no_docstring() -> None:
        """Verify missing docstring detection."""
        source = """
def undocumented():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.has_docstring is False

    @staticmethod
    def test_async_function() -> None:
        """Verify async function detection."""
        source = """
async def async_func():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.is_async is True

    @staticmethod
    def test_decorator_count() -> None:
        """Verify decorators are counted."""
        source = """
@staticmethod
@property
def decorated():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        expected_decorators = 2
        assert metrics.decorator_count == expected_decorators


class TestComplexityBuckets:
    """Tests for complexity bucket classification."""

    @staticmethod
    def test_low_complexity_bucket() -> None:
        """Verify low complexity bucket."""
        source = """
def simple():
    return 1
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        assert metrics.complexity_bucket == "low"

    @staticmethod
    def test_medium_complexity_bucket() -> None:
        """Verify medium complexity bucket."""
        # Create function with complexity > 5 but <= 10
        source = """
def medium(a, b, c, d, e, f):
    if a:
        return 1
    if b:
        return 2
    if c:
        return 3
    if d:
        return 4
    if e:
        return 5
    return 0
"""
        func = _parse_function(source)
        metrics = compute_complexity(func)
        # Should be around 6
        assert metrics.cyclomatic > COMPLEXITY_LOW
        if metrics.cyclomatic <= COMPLEXITY_MEDIUM:
            assert metrics.complexity_bucket == "medium"

    @staticmethod
    def test_complexity_thresholds_defined() -> None:
        """Verify complexity thresholds are properly defined."""
        expected_low = 5
        expected_medium = 10
        assert expected_low == COMPLEXITY_LOW
        assert expected_medium == COMPLEXITY_MEDIUM
