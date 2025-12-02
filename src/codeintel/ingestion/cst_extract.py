"""CST extraction facade.

This module re-exports the CST extraction functionality for backward compatibility
with imports that expect `codeintel.ingestion.cst_extract`.
"""

from __future__ import annotations

from codeintel.ingestion.steps.cst_extract import (
    CST_CAPTURE_CONFIG,
    CstExtractStep,
    CstVisitor,
    ModuleCstResult,
)

__all__ = [
    "CST_CAPTURE_CONFIG",
    "CstExtractStep",
    "CstVisitor",
    "ModuleCstResult",
]
