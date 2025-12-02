"""Pure computation of cyclomatic complexity and nesting depth.

This module provides functions to compute structural complexity metrics
from Python AST nodes. All functions are pure (no I/O, no side effects).

Examples
--------
>>> import ast
>>> source = '''
... def example(x):
...     if x > 0:
...         return x
...     return -x
... '''
>>> tree = ast.parse(source)
>>> func = tree.body[0]
>>> metrics = compute_complexity(func)
>>> metrics.cyclomatic
2
>>> metrics.complexity_bucket
'low'
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

COMPLEXITY_LOW: Final[int] = 5
COMPLEXITY_MEDIUM: Final[int] = 10


@dataclass(frozen=True)
class ComplexityMetrics:
    """Immutable container for function complexity metrics.

    Attributes
    ----------
    cyclomatic
        McCabe cyclomatic complexity (number of linearly independent paths).
    max_nesting_depth
        Maximum nesting level of control structures.
    return_count
        Number of return statements.
    yield_count
        Number of yield/yield from statements.
    raise_count
        Number of raise statements.
    stmt_count
        Number of statements in the function body.
    decorator_count
        Number of decorators applied to the function.
    has_docstring
        Whether the function has a docstring.
    is_async
        Whether this is an async function.
    is_generator
        Whether the function contains yield statements.
    complexity_bucket
        Categorical complexity level: "low", "medium", or "high".
    """

    cyclomatic: int
    max_nesting_depth: int
    return_count: int
    yield_count: int
    raise_count: int
    stmt_count: int
    decorator_count: int
    has_docstring: bool
    is_async: bool
    is_generator: bool
    complexity_bucket: str


class _ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that accumulates complexity metrics.

    This visitor walks a function's AST subtree and counts control flow
    constructs to compute cyclomatic complexity and nesting depth.
    """

    def __init__(self) -> None:
        """Initialize the visitor with zero counts."""
        self.return_count: int = 0
        self.yield_count: int = 0
        self.raise_count: int = 0
        self.complexity: int = 1
        self.max_nesting_depth: int = 0
        self._depth: int = 0

    def _enter_block(self) -> None:
        """Record entry into a nested block."""
        self._depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self._depth)

    def _leave_block(self) -> None:
        """Record exit from a nested block."""
        self._depth -= 1

    def visit_Return(self, node: ast.Return) -> None:
        """Count return statements."""
        del node
        self.return_count += 1

    def visit_Yield(self, node: ast.Yield) -> None:
        """Count yield statements."""
        del node
        self.yield_count += 1

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Count yield from statements."""
        del node
        self.yield_count += 1

    def visit_Raise(self, node: ast.Raise) -> None:
        """Count raise statements."""
        del node
        self.raise_count += 1

    def visit_If(self, node: ast.If) -> None:
        """Count if statements as complexity contributors."""
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_For(self, node: ast.For) -> None:
        """Count for loops as complexity contributors."""
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_While(self, node: ast.While) -> None:
        """Count while loops as complexity contributors."""
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_Try(self, node: ast.Try) -> None:
        """Count try blocks as complexity contributors."""
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_With(self, node: ast.With) -> None:
        """Count with statements as complexity contributors."""
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Count ternary expressions as complexity contributors."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Count boolean operators (and/or) as complexity contributors."""
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit nested function definitions without counting them."""
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit nested async function definitions without counting them."""
        self.generic_visit(node)


def _classify_complexity(cyclomatic: int) -> str:
    """Classify cyclomatic complexity into buckets.

    Parameters
    ----------
    cyclomatic
        The cyclomatic complexity value.

    Returns
    -------
    str
        One of "low", "medium", or "high".
    """
    if cyclomatic <= COMPLEXITY_LOW:
        return "low"
    if cyclomatic <= COMPLEXITY_MEDIUM:
        return "medium"
    return "high"


def compute_complexity(node: ast.AST) -> ComplexityMetrics:
    """Compute complexity metrics for a function AST node.

    Analyze a function or async function definition node to extract
    structural complexity metrics including cyclomatic complexity,
    nesting depth, and various statement counts.

    Parameters
    ----------
    node
        An AST node, expected to be FunctionDef or AsyncFunctionDef.
        Other node types return zeroed metrics.

    Returns
    -------
    ComplexityMetrics
        Immutable container with all computed metrics.

    Examples
    --------
    >>> import ast
    >>> source = "def f(x): return x if x > 0 else -x"
    >>> func = ast.parse(source).body[0]
    >>> m = compute_complexity(func)
    >>> m.cyclomatic
    2
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ComplexityMetrics(
            cyclomatic=0,
            max_nesting_depth=0,
            return_count=0,
            yield_count=0,
            raise_count=0,
            stmt_count=0,
            decorator_count=0,
            has_docstring=False,
            is_async=False,
            is_generator=False,
            complexity_bucket="low",
        )

    visitor = _ComplexityVisitor()
    visitor.visit(node)

    is_async = isinstance(node, ast.AsyncFunctionDef)
    is_generator = visitor.yield_count > 0
    stmt_count = len(getattr(node, "body", []))
    decorator_count = len(getattr(node, "decorator_list", []))
    has_docstring = ast.get_docstring(node) is not None
    complexity_bucket = _classify_complexity(visitor.complexity)

    return ComplexityMetrics(
        cyclomatic=visitor.complexity,
        max_nesting_depth=visitor.max_nesting_depth,
        return_count=visitor.return_count,
        yield_count=visitor.yield_count,
        raise_count=visitor.raise_count,
        stmt_count=stmt_count,
        decorator_count=decorator_count,
        has_docstring=has_docstring,
        is_async=is_async,
        is_generator=is_generator,
        complexity_bucket=complexity_bucket,
    )


__all__ = [
    "COMPLEXITY_LOW",
    "COMPLEXITY_MEDIUM",
    "ComplexityMetrics",
    "compute_complexity",
]
