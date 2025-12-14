"""Unified parsing types for CodeIntel.

This module provides the canonical parsing types used across analytics
and ingestion modules.

Examples
--------
>>> from codeintel.core.parsing import SourceSpan, AstSpanIndex
>>> import ast
>>> tree = ast.parse("def foo(): pass")
>>> index = AstSpanIndex.from_tree(tree, (ast.FunctionDef, ast.AsyncFunctionDef))
"""

from __future__ import annotations

from codeintel.core.parsing.ast_index import AstSpanIndex
from codeintel.core.parsing.models import ParsedFunction, ParsedModule
from codeintel.core.parsing.source_span import SourceSpan

__all__ = [
    "AstSpanIndex",
    "ParsedFunction",
    "ParsedModule",
    "SourceSpan",
]
