"""HTTP-facing shim around the shared backend dataset helpers.

Preferred import path for new code:
    `from codeintel.serving.backend.datasets import ...`.
"""

from __future__ import annotations

from codeintel.serving.backend.datasets import (
    DOCS_VIEWS,
    PREVIEW_COLUMN_COUNT,
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)

__all__ = [
    "DOCS_VIEWS",
    "PREVIEW_COLUMN_COUNT",
    "build_registry_and_limits",
    "describe_dataset",
    "validate_dataset_registry",
]
