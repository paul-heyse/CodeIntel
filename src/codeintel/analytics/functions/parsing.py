"""Adapters exposing the centralized parsing subsystem to function analytics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from codeintel.analytics.parsing import (
    FunctionParserRegistry,
    ParsedFunction,
    ParsedModule,
    parse_python_module,
    resolve_span,
)
from codeintel.analytics.parsing.registry import get_parser
from codeintel.config.parser_types import FunctionParserKind


def parse_functions_in_module(
    path: Path,
    content: bytes,
    *,
    kind: FunctionParserKind = FunctionParserKind.PYTHON,
) -> Iterable[ParsedFunction]:
    """
    Parse a module into ParsedFunction objects using the configured parser.

    Parameters
    ----------
    path :
        Path to the module being parsed.
    content :
        Raw file contents.
    kind :
        Parser kind to use; defaults to Python.
    """
    parser = get_parser(kind)
    return parser(path, content)


def parse_python_file(path: Path) -> ParsedModule:
    """
    Parse a Python module and return the full ParsedModule payload.

    This helper exists for analytics flows that need both functions and the
    associated span index or source lines.
    """
    content = path.read_bytes()
    return parse_python_module(path, content)


__all__ = [
    "FunctionParserRegistry",
    "ParsedFunction",
    "ParsedModule",
    "FunctionParserKind",
    "parse_functions_in_module",
    "parse_python_file",
    "resolve_span",
]
