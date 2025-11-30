"""Semantic roles analytics plugins package."""

from codeintel.analytics.semantic_roles.core import (
    FunctionContext,
    classify_function_role,
    compute_semantic_roles,
)

__all__ = [
    "FunctionContext",
    "classify_function_role",
    "compute_semantic_roles",
]
