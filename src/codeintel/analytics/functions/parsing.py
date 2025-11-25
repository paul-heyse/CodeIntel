"""Thin adapters for function parsing and span resolution."""

from __future__ import annotations

from codeintel.analytics.function_parsing import (
    FunctionParserRegistry,
    ParsedFile,
    ParsedFileLoader,
    get_parsed_file,
)
from codeintel.analytics.span_resolver import resolve_span

__all__ = [
    "FunctionParserRegistry",
    "ParsedFile",
    "ParsedFileLoader",
    "get_parsed_file",
    "resolve_span",
]
