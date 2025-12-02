"""Python AST extraction facade.

This module re-exports the AST extraction functionality for backward compatibility
with imports that expect `codeintel.ingestion.py_ast_extract`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.ingestion.steps.ast_extract import (
    AstExtractStep,
    AstMetrics,
    AstRowInfo,
    AstVisitor,
    ModuleAstResult,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def ingest_python_ast(
    gateway: StorageGateway,
    *args: object,
    **kwargs: object,
) -> None:
    """Ingest Python AST data from repository.

    This is a placeholder function for backward compatibility.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    args
        Additional positional arguments.
    kwargs
        Additional keyword arguments.
    """


__all__ = [
    "AstExtractStep",
    "AstMetrics",
    "AstRowInfo",
    "AstVisitor",
    "ModuleAstResult",
    "ingest_python_ast",
]
