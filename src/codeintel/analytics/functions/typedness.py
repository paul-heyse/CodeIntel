"""Typedness utilities for function analytics."""

from __future__ import annotations

import ast
from dataclasses import dataclass

SKIP_PARAM_NAMES = {"self", "cls"}


@dataclass(frozen=True)
class ParamStats:
    """Parameter and return annotation statistics for a function."""

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
class TypednessFlags:
    """Typedness summary flags derived from annotations."""

    param_typed_ratio: float
    unannotated_params: int
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str
    typedness_source: str


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (TypeError, ValueError, AttributeError):
        return getattr(node, "id", None) or type(node).__name__


def compute_param_stats(node: ast.AST) -> ParamStats:
    """
    Compute parameter statistics and annotations for a function node.

    Returns
    -------
    ParamStats
        Parameter counts, annotation coverage, and return annotation details.
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

    has_return_annotation = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
        node.returns is not None
    )
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


def compute_typedness_flags(
    *, total_params: int, annotated_params: int, has_return_annotation: bool
) -> TypednessFlags:
    """
    Summarize typedness coverage based on parameter and return annotations.

    Returns
    -------
    TypednessFlags
        Flags capturing coverage ratios and typedness bucket classification.
    """
    param_typed_ratio = annotated_params / float(total_params) if total_params else 1.0
    unannotated_params = total_params - annotated_params
    fully_typed = total_params > 0 and annotated_params == total_params and has_return_annotation
    untyped = annotated_params == 0 and not has_return_annotation
    partial_typed = (
        not fully_typed and not untyped and (annotated_params > 0 or has_return_annotation)
    )
    typedness_bucket = "typed" if fully_typed else "partial" if partial_typed else "untyped"
    typedness_source = (
        "annotations" if (annotated_params > 0 or has_return_annotation) else "unknown"
    )
    return TypednessFlags(
        param_typed_ratio=param_typed_ratio,
        unannotated_params=unannotated_params,
        fully_typed=fully_typed,
        partial_typed=partial_typed,
        untyped=untyped,
        typedness_bucket=typedness_bucket,
        typedness_source=typedness_source,
    )
