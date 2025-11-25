"""Central parsing and span resolution utilities for analytics."""

from __future__ import annotations

from codeintel.analytics.parsing.function_parsing import parse_python_module
from codeintel.analytics.parsing.models import ParsedFunction, ParsedModule, SourceSpan
from codeintel.analytics.parsing.registry import FunctionParserRegistry, get_parser, register_parser
from codeintel.analytics.parsing.span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)
from codeintel.analytics.parsing.validation import (
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
