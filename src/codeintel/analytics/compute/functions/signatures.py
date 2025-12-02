"""Pure computation of function signature extraction.

This module provides functions to extract structured signature information
from Python function AST nodes. All functions are pure.

Examples
--------
>>> import ast
>>> source = "async def fetch(url: str, *, timeout: float = 30.0) -> bytes: pass"
>>> func = ast.parse(source).body[0]
>>> sig = extract_signature(func)
>>> sig.is_async
True
>>> sig.name
'fetch'
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterInfo:
    """Information about a single function parameter.

    Attributes
    ----------
    name
        Parameter name.
    annotation
        String representation of type annotation, if present.
    has_default
        Whether the parameter has a default value.
    kind
        Parameter kind: "positional_only", "positional_or_keyword",
        "keyword_only", "var_positional", or "var_keyword".
    """

    name: str
    annotation: str | None
    has_default: bool
    kind: str


@dataclass(frozen=True)
class FunctionSignature:
    """Complete signature information for a function.

    Attributes
    ----------
    name
        Function name.
    qualname
        Qualified name if available.
    is_async
        Whether this is an async function.
    is_method
        Whether this appears to be a method (has self/cls first param).
    is_classmethod
        Whether decorated with @classmethod.
    is_staticmethod
        Whether decorated with @staticmethod.
    is_property
        Whether decorated with @property.
    parameters
        Tuple of parameter information.
    return_annotation
        String representation of return type, if present.
    decorators
        Tuple of decorator names.
    docstring
        Function docstring, if present.
    """

    name: str
    qualname: str
    is_async: bool
    is_method: bool
    is_classmethod: bool
    is_staticmethod: bool
    is_property: bool
    parameters: tuple[ParameterInfo, ...]
    return_annotation: str | None
    decorators: tuple[str, ...]
    docstring: str | None


def _annotation_to_str(node: ast.AST | None) -> str | None:
    """Convert an annotation AST node to string.

    Returns
    -------
    str | None
        String representation of the annotation, or None.
    """
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (TypeError, ValueError, AttributeError):
        return getattr(node, "id", None) or type(node).__name__


def _decorator_name(node: ast.expr) -> str:
    """Extract decorator name from decorator node.

    Returns
    -------
    str
        Decorator name as string.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except (TypeError, ValueError):
            return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    try:
        return ast.unparse(node)
    except (TypeError, ValueError):
        return type(node).__name__


def _extract_parameters(args: ast.arguments) -> tuple[ParameterInfo, ...]:
    """Extract parameter information from function arguments.

    Returns
    -------
    tuple[ParameterInfo, ...]
        Extracted parameter information.
    """
    params: list[ParameterInfo] = []

    # Positional-only parameters (Python 3.8+)
    posonlyargs = getattr(args, "posonlyargs", [])
    num_posonly_defaults = len(args.defaults) - (len(args.args) - len(posonlyargs))
    posonly_defaults_start = max(0, len(posonlyargs) - max(0, num_posonly_defaults))

    for i, param in enumerate(posonlyargs):
        has_default = i >= posonly_defaults_start
        params.append(
            ParameterInfo(
                name=param.arg,
                annotation=_annotation_to_str(param.annotation),
                has_default=has_default,
                kind="positional_only",
            )
        )

    # Regular positional/keyword parameters
    num_args_defaults = len(args.defaults) - max(0, num_posonly_defaults)
    args_defaults_start = len(args.args) - num_args_defaults

    for i, param in enumerate(args.args):
        has_default = i >= args_defaults_start
        params.append(
            ParameterInfo(
                name=param.arg,
                annotation=_annotation_to_str(param.annotation),
                has_default=has_default,
                kind="positional_or_keyword",
            )
        )

    # *args
    if args.vararg is not None:
        params.append(
            ParameterInfo(
                name=args.vararg.arg,
                annotation=_annotation_to_str(args.vararg.annotation),
                has_default=False,
                kind="var_positional",
            )
        )

    # Keyword-only parameters
    for i, param in enumerate(args.kwonlyargs):
        default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        params.append(
            ParameterInfo(
                name=param.arg,
                annotation=_annotation_to_str(param.annotation),
                has_default=default is not None,
                kind="keyword_only",
            )
        )

    # **kwargs
    if args.kwarg is not None:
        params.append(
            ParameterInfo(
                name=args.kwarg.arg,
                annotation=_annotation_to_str(args.kwarg.annotation),
                has_default=False,
                kind="var_keyword",
            )
        )

    return tuple(params)


def extract_signature(
    node: ast.AST,
    *,
    qualname: str | None = None,
) -> FunctionSignature:
    r"""Extract complete signature information from a function AST node.

    Analyze a function definition to extract its full signature including
    parameters, decorators, return type, and other metadata.

    Parameters
    ----------
    node
        An AST node, expected to be FunctionDef or AsyncFunctionDef.
        Other node types return an empty signature.
    qualname
        Optional qualified name to use. Defaults to the function name.

    Returns
    -------
    FunctionSignature
        Complete signature information for the function.

    Examples
    --------
    >>> import ast
    >>> source = "@decorator\ndef greet(name: str) -> str: '''Say hello.'''\n    pass"
    >>> func = ast.parse(source).body[0]
    >>> sig = extract_signature(func)
    >>> sig.name
    'greet'
    >>> sig.docstring
    'Say hello.'
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return FunctionSignature(
            name="",
            qualname="",
            is_async=False,
            is_method=False,
            is_classmethod=False,
            is_staticmethod=False,
            is_property=False,
            parameters=(),
            return_annotation=None,
            decorators=(),
            docstring=None,
        )

    name = node.name
    resolved_qualname = qualname or name
    is_async = isinstance(node, ast.AsyncFunctionDef)

    # Extract decorators
    decorators = tuple(_decorator_name(d) for d in node.decorator_list)
    decorator_names = {d.split(".")[0] for d in decorators}

    is_classmethod = "classmethod" in decorator_names
    is_staticmethod = "staticmethod" in decorator_names
    is_property = "property" in decorator_names

    # Extract parameters
    parameters = _extract_parameters(node.args)

    # Determine if method (has self/cls as first param)
    is_method = False
    if parameters and not is_staticmethod:
        first_param = parameters[0].name
        is_method = first_param in {"self", "cls"}

    return_annotation = _annotation_to_str(node.returns)
    docstring = ast.get_docstring(node)

    return FunctionSignature(
        name=name,
        qualname=resolved_qualname,
        is_async=is_async,
        is_method=is_method,
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
        is_property=is_property,
        parameters=parameters,
        return_annotation=return_annotation,
        decorators=decorators,
        docstring=docstring,
    )


__all__ = [
    "FunctionSignature",
    "ParameterInfo",
    "extract_signature",
]
