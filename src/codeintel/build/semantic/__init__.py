"""Semantic registry compilation helpers for build targets."""

from codeintel.build.semantic.registry_compiler import (
    CompiledSemanticRegistry,
    SemanticTagIssue,
    SemanticTagValidationError,
    compile_semantic_registry,
    compile_semantic_registry_from_views,
)

__all__ = [
    "CompiledSemanticRegistry",
    "SemanticTagIssue",
    "SemanticTagValidationError",
    "compile_semantic_registry",
    "compile_semantic_registry_from_views",
]
