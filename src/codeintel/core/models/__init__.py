"""Unified model infrastructure.

This module provides core model patterns for the codebase,
including row protocols and base types.
"""

from codeintel.core.models.rows import (
    RowModelProtocol,
    RowType,
)

__all__ = [
    "RowModelProtocol",
    "RowType",
]
