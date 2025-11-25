"""Concrete parser implementations wired into the analytics registry."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from codeintel.analytics.parsing.models import ParsedFunction, ParsedModule, SourceSpan
from codeintel.analytics.parsing.registry import register_parser
from codeintel.config.parser_types import FunctionParserKind
from codeintel.ingestion.ast_utils import AstSpanIndex


def _qualname(parts: list[str], name: str) -> str:
    return ".".join([*parts, name])


def _span_for_node(path: Path, node: ast.AST) -> SourceSpan:
    start_line = int(getattr(node, "lineno", 0) or 0)
    end_line = int(getattr(node, "end_lineno", start_line) or start_line)
    start_col = int(getattr(node, "col_offset", 0) or 0)
    end_col = int(getattr(node, "end_col_offset", start_col) or start_col)
    return SourceSpan(
        path=path,
        start_line=start_line,
        start_col=start_col,
        end_line=end_line,
        end_col=end_col,
    )


def _is_any_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id == "Any"
    if isinstance(node, ast.Attribute):
        return node.attr == "Any"
    return False


def _collect_param_annotations(args: ast.arguments) -> tuple[dict[str, Any], dict[str, bool]]:
    annotations: dict[str, Any] = {}
    any_flags: dict[str, bool] = {}

    def _record(arg: ast.arg | None) -> None:
        if arg is None:
            return
        annotations[arg.arg] = arg.annotation
        any_flags[arg.arg] = _is_any_annotation(arg.annotation)

    for arg in args.posonlyargs:
        _record(arg)
    for arg in args.args:
        _record(arg)
    _record(args.vararg)
    for arg in args.kwonlyargs:
        _record(arg)
    _record(args.kwarg)
    return annotations, any_flags


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._stack: list[str] = []
        self.functions: list[ParsedFunction] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Capture function definitions."""
        self._record_function(node)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Capture async function definitions."""
        self._record_function(node)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        span = _span_for_node(self._path, node)
        param_annotations, any_flags = _collect_param_annotations(node.args)
        parsed = ParsedFunction(
            path=self._path,
            qualname=_qualname(self._stack, node.name),
            function_goid_h128=None,
            span=span,
            ast=node,
            docstring=ast.get_docstring(node),
            param_annotations=param_annotations,
            return_annotation=node.returns,
            param_any_flags=any_flags,
            return_is_any=_is_any_annotation(node.returns),
        )
        self.functions.append(parsed)


def parse_python_module(path: Path, content: bytes) -> ParsedModule:
    """
    Parse a Python module and extract function spans and annotations.

    Parameters
    ----------
    path :
        Absolute path to the module being parsed.
    content :
        Raw file contents.

    Returns
    -------
    ParsedModule
        Parsed source, split lines, and extracted functions.

    Raises
    ------
    ValueError
        If the module cannot be decoded or parsed.
    """
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        message = f"Failed to decode {path}"
        raise ValueError(message) from exc
    try:
        module_ast = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        message = f"Failed to parse {path}: {exc.msg}"
        raise ValueError(message) from exc

    collector = _FunctionCollector(path)
    collector.visit(module_ast)
    span_index = AstSpanIndex.from_tree(module_ast, (ast.FunctionDef, ast.AsyncFunctionDef))
    lines = source.splitlines()
    return ParsedModule(
        path=path,
        source=source,
        lines=lines,
        module_ast=module_ast,
        span_index=span_index,
        functions=collector.functions,
    )


def _parse_python_functions(path: Path, content: bytes) -> Iterable[ParsedFunction]:
    return parse_python_module(path, content).functions


register_parser(FunctionParserKind.PYTHON, _parse_python_functions)
