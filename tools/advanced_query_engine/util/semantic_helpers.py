"""Semantic analysis helpers for query handlers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from tools.advanced_query_engine.contracts import Span
from tools.advanced_query_engine.util.line_index import LineIndex

PATH_KIND_TEST = "test"
PATH_KIND_DOC = "doc"
PATH_KIND_EXAMPLE = "example"
PATH_KIND_PROD = "prod"

_DECORATOR_RE = re.compile(r"^@([A-Za-z_][A-Za-z0-9_\.]*)")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass(frozen=True)
class SignatureInfo:
    """Parsed signature metadata."""

    positional: list[str]
    kwonly: list[str]
    vararg: str | None
    kwarg: str | None


@dataclass(frozen=True)
class CallArgs:
    """Parsed call argument metadata."""

    positional: list[str]
    keywords: dict[str, str]
    has_vararg: bool
    has_kwarg: bool


def classify_path_kind(path: str) -> str:
    """Classify a path as prod/test/doc/example.

    Parameters
    ----------
    path:
        Repo-relative path.

    Returns
    -------
    str
        One of PATH_KIND_* constants.
    """
    normalized = path.replace("\\", "/").lower()
    filename = PurePosixPath(normalized).name
    if "/tests/" in normalized or filename.startswith("test_") or filename.endswith("_test.py"):
        return PATH_KIND_TEST
    if "/docs/" in normalized or filename.startswith("readme"):
        return PATH_KIND_DOC
    if "/examples/" in normalized or "/example/" in normalized:
        return PATH_KIND_EXAMPLE
    return PATH_KIND_PROD


def module_qname_from_path(path: str) -> str:
    """Convert a repo-relative path into a dotted module name.

    Parameters
    ----------
    path:
        Repo-relative path.

    Returns
    -------
    str
        Dotted module name.
    """
    pure = PurePosixPath(path)
    if pure.suffix == ".py":
        pure = pure.with_suffix("")
    if pure.name == "__init__":
        pure = pure.parent
    return ".".join(part for part in pure.parts if part)


def package_prefix(path: str, *, depth: int) -> str:
    """Return a package prefix for a path at the given depth.

    Parameters
    ----------
    path:
        Repo-relative path.
    depth:
        Number of path components to include.

    Returns
    -------
    str
        Package prefix string.
    """
    if depth <= 0:
        return ""
    parts = PurePosixPath(path).parts
    return "/".join(parts[:depth])


def extract_decorators(source: bytes, line_index: LineIndex, span: Span) -> list[str]:
    """Extract decorator names immediately above a definition span.

    Parameters
    ----------
    source:
        Source bytes for the file.
    line_index:
        Line index for the source.
    span:
        Definition span to inspect.

    Returns
    -------
    list[str]
        Decorator names in source order.
    """
    start_line, _ = line_index.line_col(span.start_byte)
    decorators: list[str] = []
    line_no = start_line - 1
    while line_no >= 1:
        begin = line_index.line_start_byte(line_no)
        finish = line_index.line_start_byte(line_no + 1)
        text = source[begin:finish].decode("utf-8", errors="replace").strip()
        if not text:
            break
        if not text.startswith("@"):
            break
        match = _DECORATOR_RE.match(text)
        if match:
            decorators.append(match.group(1))
        line_no -= 1
    decorators.reverse()
    return decorators


def tokenize_words(text: str | None) -> set[str]:
    """Return lowercase word tokens for a snippet of text.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    set[str]
        Unique word tokens.
    """
    if not text:
        return set()
    return {token.lower() for token in _WORD_RE.findall(text)}


def parse_signature(signature: str | None) -> SignatureInfo | None:
    """Parse a function signature string into parameter metadata.

    Parameters
    ----------
    signature:
        Signature string like ``def foo(a, b)``.

    Returns
    -------
    SignatureInfo | None
        Parsed signature metadata or None on failure.
    """
    if not signature:
        return None
    source = f"{signature}:\n    return None"
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    if not module.body or not isinstance(module.body[0], ast.FunctionDef):
        return None
    fn = module.body[0]
    positional = [arg.arg for arg in fn.args.posonlyargs + fn.args.args]
    kwonly = [arg.arg for arg in fn.args.kwonlyargs]
    vararg = fn.args.vararg.arg if fn.args.vararg else None
    kwarg = fn.args.kwarg.arg if fn.args.kwarg else None
    return SignatureInfo(
        positional=positional,
        kwonly=kwonly,
        vararg=vararg,
        kwarg=kwarg,
    )


def parse_call_args(call_expr: str) -> CallArgs | None:
    """Parse a call expression into positional/keyword arguments.

    Parameters
    ----------
    call_expr:
        Call expression string (e.g., ``foo(a, b=1)``).

    Returns
    -------
    CallArgs | None
        Parsed arguments or None when parsing fails.
    """
    try:
        module = ast.parse(call_expr)
    except SyntaxError:
        return None
    if not module.body or not isinstance(module.body[0], ast.Expr):
        return None
    expr = module.body[0].value
    if not isinstance(expr, ast.Call):
        return None
    positional = [ast.unparse(arg) for arg in expr.args]
    keywords: dict[str, str] = {}
    has_vararg = False
    has_kwarg = False
    for keyword in expr.keywords:
        if keyword.arg is None:
            if isinstance(keyword.value, ast.Dict):
                has_kwarg = True
            else:
                has_vararg = True
            continue
        keywords[keyword.arg] = ast.unparse(keyword.value)
    return CallArgs(
        positional=positional,
        keywords=keywords,
        has_vararg=has_vararg,
        has_kwarg=has_kwarg,
    )


def map_args_to_params(signature: SignatureInfo, args: CallArgs) -> dict[str, str]:
    """Map call arguments to parameters when possible.

    Parameters
    ----------
    signature:
        Parsed signature metadata.
    args:
        Parsed call arguments.

    Returns
    -------
    dict[str, str]
        Mapping of parameter names to argument expressions.
    """
    mapping: dict[str, str] = {}
    pos_params = list(signature.positional)
    kwonly_params = list(signature.kwonly)
    for idx, arg_value in enumerate(args.positional):
        if idx < len(pos_params):
            mapping[pos_params[idx]] = arg_value
        elif signature.vararg:
            mapping[f"*{signature.vararg}"] = arg_value
    for key, value in args.keywords.items():
        if key in pos_params or key in kwonly_params:
            mapping[key] = value
        elif signature.kwarg:
            mapping[f"**{signature.kwarg}"] = value
    return mapping


def callee_label(expr: str) -> str:
    """Return the terminal name for a callee expression.

    Parameters
    ----------
    expr:
        Callee expression text.

    Returns
    -------
    str
        Last identifier segment.
    """
    text = expr.strip()
    if not text:
        return ""
    return text.split(".")[-1]


__all__ = [
    "PATH_KIND_DOC",
    "PATH_KIND_EXAMPLE",
    "PATH_KIND_PROD",
    "PATH_KIND_TEST",
    "CallArgs",
    "SignatureInfo",
    "callee_label",
    "classify_path_kind",
    "extract_decorators",
    "map_args_to_params",
    "module_qname_from_path",
    "package_prefix",
    "parse_call_args",
    "parse_signature",
    "tokenize_words",
]
