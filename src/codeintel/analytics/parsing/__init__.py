"""Central parsing and span resolution utilities for analytics."""

from __future__ import annotations

from .function_parsing import parse_python_module
from .models import ParsedFunction, ParsedModule, SourceSpan
from .registry import FunctionParserRegistry, get_parser, register_parser
from .span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)
from .validation import (
    BaseValidationReporter,
    FunctionValidationReporter,
    GraphValidationReporter,
)

__all__ = [
    "BaseValidationReporter",
    "FunctionParserRegistry",
    "FunctionValidationReporter",
    "GraphValidationReporter",
    "ParsedFunction",
    "ParsedModule",
    "SourceSpan",
    "SpanResolutionError",
    "SpanResolutionResult",
    "build_span_index",
    "get_parser",
    "parse_python_module",
    "register_parser",
    "resolve_span",
]
