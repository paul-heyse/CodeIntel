"""LibCST parsing adapter implementing ParsingPort.

This module provides a concrete implementation of ParsingPort that
uses LibCST for source code parsing operations.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import libcst as cst

from codeintel.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
    ParseError,
    ParseResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.graphs.ports.parsing import (
        ParsingPort,
    )

log = logging.getLogger(__name__)


@dataclass
class _LibcstCollector(cst.CSTVisitor):
    """Single-pass collector for imports and functions using libcst hooks."""

    imports: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    functions: list[ParsedFunction] = field(default_factory=list)
    _class_stack: list[str] = field(default_factory=list, repr=False)

    def on_visit(self, node: cst.CSTNode) -> bool:
        """Process nodes using libcst generic visit hooks.

        Returns
        -------
        bool
            True to continue traversal.
        """
        if isinstance(node, cst.ClassDef):
            self._class_stack.append(node.name.value)
        elif isinstance(node, cst.FunctionDef):
            self._handle_function(node)
        elif isinstance(node, cst.Import):
            self._handle_import(node)
        elif isinstance(node, cst.ImportFrom):
            self._handle_import_from(node)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Track class nesting when leaving nodes."""
        if isinstance(original_node, cst.ClassDef) and self._class_stack:
            self._class_stack.pop()

    def _handle_function(self, node: cst.FunctionDef) -> None:
        name = node.name.value
        qualname = ".".join([*self._class_stack, name])

        is_async = node.asynchronous is not None

        decorators = [
            dec_name
            for dec in node.decorators
            if (dec_name := _get_decorator_name(dec.decorator)) is not None
        ]

        params = [
            param.name.value for param in node.params.params if isinstance(param.name, cst.Name)
        ]

        self.functions.append(
            ParsedFunction(
                name=name,
                qualname=qualname,
                start_line=1,
                end_line=1,
                is_async=is_async,
                decorator_names=tuple(decorators),
                parameters=tuple(params),
            )
        )

    def _handle_import(self, node: cst.Import) -> None:
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                module_name = _get_module_name(alias.name)
                if not module_name:
                    continue
                local_name = (
                    alias.asname.name.value
                    if alias.asname and isinstance(alias.asname.name, cst.Name)
                    else module_name.split(".")[-1]
                )
                self.imports.append((module_name, (local_name,)))
                self.aliases[local_name] = module_name

    def _handle_import_from(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        module_name = _get_module_name(node.module)
        if not module_name:
            return

        if isinstance(node.names, cst.ImportStar):
            self.imports.append((module_name, ("*",)))
            return

        names: list[str] = []
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                name_str = _get_name_value(alias.name)
                if not name_str:
                    continue
                local_name = (
                    alias.asname.name.value
                    if alias.asname and isinstance(alias.asname.name, cst.Name)
                    else name_str
                )
                names.append(local_name)
                self.aliases[local_name] = f"{module_name}.{name_str}"
        if names:
            self.imports.append((module_name, tuple(names)))


def _get_module_name(node: cst.BaseExpression) -> str | None:
    """Extract module name from a CST expression.

    Parameters
    ----------
    node
        CST expression representing a module name.

    Returns
    -------
    str | None
        Module name string or None if extraction fails.
    """
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _get_module_name(node.value)
        if base:
            return f"{base}.{node.attr.value}"
    return None


def _get_name_value(node: cst.BaseExpression) -> str | None:
    """Extract name value from a CST expression.

    Parameters
    ----------
    node
        CST expression.

    Returns
    -------
    str | None
        Name string or None.
    """
    if isinstance(node, cst.Name):
        return node.value
    return None


def _get_decorator_name(node: cst.BaseExpression) -> str | None:
    """Extract decorator name from a CST expression.

    Parameters
    ----------
    node
        Decorator expression.

    Returns
    -------
    str | None
        Decorator name or None.
    """
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return _get_module_name(node)
    if isinstance(node, cst.Call):
        return _get_decorator_name(node.func)
    return None


@dataclass
class LibCSTParsingAdapter:
    """ParsingPort implementation using LibCST.

    This adapter parses Python source code using LibCST and provides
    structured representations for use by pure computation functions.
    """

    @staticmethod
    def parse_module(source: str) -> ParseResult:
        """Parse a module from source code.

        Parameters
        ----------
        source
            Python source code string.

        Returns
        -------
        ParseResult
            Parsed module or error.
        """
        try:
            cst_module = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            return ParseResult.fail(
                ParseError(
                    message=str(exc),
                    line=getattr(exc, "lineno", None),
                    column=getattr(exc, "col_offset", None),
                )
            )

        try:
            ast_module = ast.parse(source)
        except SyntaxError:
            ast_module = None

        collector = _LibcstCollector()
        wrapper = cst.MetadataWrapper(cst_module)
        wrapper.visit(collector)

        return ParseResult.ok(
            ParsedModule(
                source=source,
                functions=tuple(collector.functions),
                imports=tuple(collector.imports),
                cst_module=cst_module,
                ast_module=ast_module,
                _import_aliases=collector.aliases,
            )
        )

    def parse_function(
        self,
        source: str,
        start_line: int,
        end_line: int,
    ) -> ParsedFunction | None:
        """Parse a single function from source within a line range.

        Parameters
        ----------
        source
            Source code string.
        start_line
            Starting line number.
        end_line
            Ending line number.

        Returns
        -------
        ParsedFunction | None
            Parsed function if found within range.
        """
        result = self.parse_module(source)
        if not result.success or result.module is None:
            return None

        for func in result.module.functions:
            if func.start_line >= start_line and func.end_line <= end_line:
                return func
        return None

    @staticmethod
    def extract_imports(source: str) -> Sequence[tuple[str, tuple[str, ...]]]:
        """Extract import statements from source.

        Parameters
        ----------
        source
            Source code string.

        Returns
        -------
        Sequence[tuple[str, tuple[str, ...]]]
            Sequence of (module, imported_names) tuples.
        """
        result = LibCSTParsingAdapter.parse_module(source)
        if not result.success or result.module is None:
            return []
        return result.module.imports

    @staticmethod
    def extract_call_sites(
        module: ParsedModule,
        function_span: tuple[int, int],
    ) -> Sequence[tuple[str, int]]:
        """Extract call sites within a function.

        Parameters
        ----------
        module
            Parsed module containing the function.
        function_span
            (start_line, end_line) of the function.

        Returns
        -------
        Sequence[tuple[str, int]]
            Sequence of (callee_name, line_number) tuples.
        """
        if module.ast_module is None:
            return []

        start_line, end_line = function_span
        call_sites: list[tuple[str, int]] = []

        for node in ast.walk(module.ast_module):
            if isinstance(node, ast.Call):
                line = getattr(node, "lineno", 0)
                if start_line <= line <= end_line:
                    callee = _extract_call_name(node.func)
                    if callee:
                        call_sites.append((callee, line))

        return call_sites


def _extract_call_name(node: ast.expr) -> str | None:
    """Extract the callee name from a call expression.

    Parameters
    ----------
    node
        AST expression for the callee.

    Returns
    -------
    str | None
        Callee name or None.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _check_protocol_compliance() -> None:
    """Verify LibCSTParsingAdapter implements ParsingPort."""
    adapter: ParsingPort = LibCSTParsingAdapter()
    _ = adapter


__all__ = ["LibCSTParsingAdapter"]
