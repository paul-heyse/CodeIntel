"""Shared Hamilton integration primitives (core-owned).

This package contains small, dependency-light utilities shared across build,
storage, and serving that relate to Hamilton tagging and execution records.

It exists to avoid layering violations (e.g., storage importing build) while
keeping the Hamilton-first architecture explicit.
"""

from __future__ import annotations

from codeintel.core.hamilton.tag_filters import (
    tf_artifacts,
    tf_datasets,
    tf_savers,
    tf_semantic_views,
)
from codeintel.core.hamilton.tag_query import TagQuery

__all__ = [
    "TagQuery",
    "tf_artifacts",
    "tf_datasets",
    "tf_savers",
    "tf_semantic_views",
]
