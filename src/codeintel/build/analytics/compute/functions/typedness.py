"""Pure computation of parameter annotation statistics.

This module provides functions to analyze type annotations on Python
function parameters and return values. All functions are pure.
The module name is retained for backward compatibility.

Examples
--------
>>> import ast
>>> source = "def greet(name: str, count: int = 1) -> str: pass"
>>> func = ast.parse(source).body[0]
>>> stats = compute_param_stats(func)
>>> stats.annotated_params
2
>>> stats.has_return_annotation
True
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

SKIP_PARAM_NAMES: Final[frozenset[str]] = frozenset({"self", "cls"})


@dataclass(frozen=True)
class ParamStats:
    """Parameter and return annotation statistics for a function.

    Attributes
    ----------
    param_count
        Total number of parameters including *args and **kwargs.
    positional_params
        Number of positional and positional-only parameters.
    keyword_only_params
        Number of keyword-only parameters.
    has_varargs
        Whether the function accepts *args.
    has_varkw
        Whether the function accepts **kwargs.
    total_params
        Parameters counted for annotation stats (excludes self/cls).
    annotated_params
        Number of parameters with type annotations.
    param_types
        Mapping of parameter names to their type annotation strings.
    has_return_annotation
        Whether the function has a return type annotation.
    return_type
        String representation of return type annotation, if present.
    """

    param_count: int
    positional_params: int
    keyword_only_params: int
    has_varargs: bool
    has_varkw: bool
    total_params: int
    annotated_params: int
    param_types: dict[str, str | None]
    has_return_annotation: bool
    return_type: str | None


@dataclass(frozen=True)
def _annotation_to_str(node: ast.AST | None) -> str | None:
    """Convert an annotation AST node to a string representation.

    Parameters
    ----------
    node
        An AST node representing a type annotation, or None.

    Returns
    -------
    str | None
        String representation of the annotation, or None if not present.
    """
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (TypeError, ValueError, AttributeError):
        return getattr(node, "id", None) or type(node).__name__


def compute_param_stats(node: ast.AST) -> ParamStats:
    """Compute parameter statistics and annotations for a function node.

    Analyze a function definition to extract counts and type annotations
    for all parameters. Parameters named 'self' or 'cls' are excluded
    from annotation statistics.

    Parameters
    ----------
    node
        An AST node, expected to be FunctionDef or AsyncFunctionDef.
        Other node types return zeroed statistics.

    Returns
    -------
    ParamStats
        Parameter counts, annotation completeness, and return annotation details.

    Examples
    --------
    >>> import ast
    >>> source = "def f(x: int, y) -> str: pass"
    >>> func = ast.parse(source).body[0]
    >>> stats = compute_param_stats(func)
    >>> stats.total_params
    2
    >>> stats.annotated_params
    1
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ParamStats(
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

    args = node.args
    all_params = list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_params.append(args.vararg)
    if args.kwarg is not None:
        all_params.append(args.kwarg)

    param_count = len(all_params)
    positional_params = len(getattr(args, "posonlyargs", [])) + len(args.args)
    keyword_only_params = len(args.kwonlyargs)
    has_varargs = args.vararg is not None
    has_varkw = args.kwarg is not None

    total_params = 0
    annotated_params = 0
    param_types: dict[str, str | None] = {}

    for param in all_params:
        name = param.arg
        if name in SKIP_PARAM_NAMES:
            continue
        total_params += 1
        ann_str = _annotation_to_str(param.annotation)
        if ann_str is not None:
            annotated_params += 1
        param_types[name] = ann_str

    has_return_annotation = node.returns is not None
    return_type = _annotation_to_str(node.returns) if hasattr(node, "returns") else None

    return ParamStats(
        param_count=param_count,
        positional_params=positional_params,
        keyword_only_params=keyword_only_params,
        has_varargs=has_varargs,
        has_varkw=has_varkw,
        total_params=total_params,
        annotated_params=annotated_params,
        param_types=param_types,
        has_return_annotation=has_return_annotation,
        return_type=return_type,
    )


__all__ = [
    "SKIP_PARAM_NAMES",
    "ParamStats",
    "compute_param_stats",
]
