"""Compatibility re-export for declared schema definitions.

The canonical declared schema definitions live in
``codeintel.config.datasets.declared_schemas``. This module preserves the historical import path
``codeintel.build.schemas.declared_schemas``.
"""

from __future__ import annotations

from codeintel.config.datasets.declared_schemas import COMPOSITE_SCHEMAS, TABLE_SCHEMAS

__all__ = [
    "COMPOSITE_SCHEMAS",
    "TABLE_SCHEMAS",
]
