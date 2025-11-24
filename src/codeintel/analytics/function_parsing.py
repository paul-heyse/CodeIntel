"""Shared parsing helpers and registry for function analytics."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codeintel.config.parser_types import FunctionParserKind
from codeintel.ingestion.ast_utils import AstSpanIndex, timed_parse

ParsedFileLoader = Callable[[Path], "ParsedFile | None"]


@dataclass(frozen=True)
class ParsedFile:
    """Parsed module contents and span index."""

    lines: list[str]
    index: AstSpanIndex


class FunctionParserRegistry:
    """Registry of available function parsers keyed by parser kind."""

    def __init__(self) -> None:
        self._parsers: dict[FunctionParserKind, ParsedFileLoader] = {
            FunctionParserKind.PYTHON: parse_python_file,
        }

    def get(self, kind: FunctionParserKind | None) -> ParsedFileLoader:
        """
        Return the parser for the given kind (defaults to python).

        Parameters
        ----------
        kind : FunctionParserKind | None
            Parser identifier; falls back to `FunctionParserKind.PYTHON` when None.

        Returns
        -------
        ParsedFileLoader
            Callable that parses a file path into a ParsedFile or returns None.

        Raises
        ------
        ValueError
            If no parser is registered for the requested kind.
        """
        target = kind or FunctionParserKind.PYTHON
        if target not in self._parsers:
            message = f"Unsupported function parser: {target}"
            raise ValueError(message)
        return self._parsers[target]

    def register(self, kind: FunctionParserKind, loader: ParsedFileLoader) -> None:
        """
        Register or replace a parser implementation.

        Parameters
        ----------
        kind:
            Parser identifier.
        loader:
            Callable that returns a ParsedFile or None when parsing fails.
        """
        self._parsers[kind] = loader


def parse_python_file(file_path: Path) -> ParsedFile | None:
    """
    Parse a Python file and build a span index for functions.

    Parameters
    ----------
    file_path : Path
        Absolute path to the Python source file to parse.

    Returns
    -------
    ParsedFile | None
        Parsed module contents with span index, or None when parsing fails.
    """
    parsed = timed_parse(file_path)
    if parsed is None:
        return None
    lines, tree, _duration = parsed
    index = AstSpanIndex.from_tree(tree, (ast.FunctionDef, ast.AsyncFunctionDef))
    return ParsedFile(lines=lines, index=index)


def get_parsed_file(
    rel_path: str,
    abs_path: Path,
    cache: dict[str, ParsedFile | None],
    parser: ParsedFileLoader,
) -> ParsedFile | None:
    """
    Load and cache a parsed module for a given relative path.

    Parameters
    ----------
    rel_path : str
        Repository-relative path used as cache key.
    abs_path : Path
        Absolute path to the target file.
    cache : dict[str, ParsedFile | None]
        Cache mapping relative paths to parsed results or None.
    parser : ParsedFileLoader
        Callable that parses the provided absolute path.

    Returns
    -------
    ParsedFile | None
        Parsed module data or None when missing/unparsable.
    """
    if rel_path in cache:
        return cache[rel_path]
    if not abs_path.is_file():
        cache[rel_path] = None
        return None
    parsed = parser(abs_path)
    cache[rel_path] = parsed
    return parsed
