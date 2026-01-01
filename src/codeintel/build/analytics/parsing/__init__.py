"""Central parsing and span resolution utilities for analytics."""

from __future__ import annotations

from codeintel.build.analytics.parsing.compute import (
    ValidationRows,
    get_validation_rows,
)
from codeintel.build.analytics.parsing.function_parsing import parse_python_module
from codeintel.build.analytics.parsing.registry import (
    FunctionParserRegistry,
    get_parser,
    register_parser,
)
from codeintel.build.analytics.parsing.span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)
from codeintel.core.parsing import ParsedFunction, ParsedModule, SourceSpan
from codeintel.core.validation.reporters import (
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
    "ValidationRows",
    "build_span_index",
    "get_parser",
    "get_validation_rows",
    "parse_python_module",
    "register_parser",
    "resolve_span",
]
