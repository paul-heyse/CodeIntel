"""Central parsing and span resolution utilities for analytics.

For Hamilton native execution, use the materialization helpers:
- `materialize_function_validation` for function validation rows
- `materialize_graph_validation` for graph validation rows
"""

from __future__ import annotations

from codeintel.analytics.parsing.compute import (
    ValidationResult,
    get_validation_rows,
    materialize_function_validation,
    materialize_graph_validation,
)
from codeintel.analytics.parsing.function_parsing import parse_python_module
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
from codeintel.core.parsing import ParsedFunction, ParsedModule, SourceSpan

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
    "ValidationResult",
    "build_span_index",
    "get_parser",
    "get_validation_rows",
    "materialize_function_validation",
    "materialize_graph_validation",
    "parse_python_module",
    "register_parser",
    "resolve_span",
]
