"""Shared AST feature models, patterns, and extraction utilities."""

from __future__ import annotations

from codeintel.build.analytics.ast_features.extract import (
    build_import_map,
    compute_function_features,
    io_flags_from_call,
)
from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.ast_features.patterns import (
    CONCURRENCY_LIBS,
    DB_LIBS,
    DEFAULT_IO_SPEC,
    DEFAULT_PATTERNS,
    HTTP_CLIENT_LIBS,
    HTTP_SERVER_LIBS,
    MESSAGE_LIBS,
    AstFeaturePatterns,
)

__all__ = [
    "CONCURRENCY_LIBS",
    "DB_LIBS",
    "DEFAULT_IO_SPEC",
    "DEFAULT_PATTERNS",
    "HTTP_CLIENT_LIBS",
    "HTTP_SERVER_LIBS",
    "MESSAGE_LIBS",
    "AstFeaturePatterns",
    "FunctionAstFeatures",
    "IoFlags",
    "build_import_map",
    "compute_function_features",
    "io_flags_from_call",
]
