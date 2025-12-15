"""Semantic serving primitives (registry, inventory, query kernel)."""

from __future__ import annotations

from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import (
    FilterSpec,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewSpec,
)
from codeintel.serving.semantic.registry import SemanticRegistry

__all__ = [
    "FilterSpec",
    "SemanticQueryKernel",
    "SemanticQueryRequest",
    "SemanticQueryResponse",
    "SemanticRegistry",
    "SemanticViewSpec",
]
