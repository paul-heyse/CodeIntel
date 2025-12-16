"""Backward compatibility alias for semantic view tagging utilities.

The canonical location for semantic tagging utilities is
`codeintel.hamilton.semantic_tags`. This module remains as a build-time import
path for existing code and external integrations.
"""

from __future__ import annotations

from codeintel.hamilton.semantic_tags import (
    SEMANTIC_VIEW_TAG_ATTR,
    TAG_DEFAULT_LIMIT,
    TAG_DEFAULT_ORDER,
    TAG_DEPRECATED,
    TAG_MCP_VISIBLE,
    TAG_OUTPUT_KIND,
    TAG_REPLACED_BY,
    TAG_SEMANTIC_COLS,
    TAG_SEMANTIC_DESC,
    TAG_SEMANTIC_ENTITY,
    TAG_SEMANTIC_GRAIN,
    TAG_SEMANTIC_ID,
    TAG_SEMANTIC_JOINS,
    TAG_SEMANTIC_KIND,
    TAG_SEMANTIC_PK,
    TAG_SENSITIVITY,
    TAG_TABLE_KEY,
    get_semantic_view_tags,
    semantic_view,
)

__all__ = [
    "SEMANTIC_VIEW_TAG_ATTR",
    "TAG_DEFAULT_LIMIT",
    "TAG_DEFAULT_ORDER",
    "TAG_DEPRECATED",
    "TAG_MCP_VISIBLE",
    "TAG_OUTPUT_KIND",
    "TAG_REPLACED_BY",
    "TAG_SEMANTIC_COLS",
    "TAG_SEMANTIC_DESC",
    "TAG_SEMANTIC_ENTITY",
    "TAG_SEMANTIC_GRAIN",
    "TAG_SEMANTIC_ID",
    "TAG_SEMANTIC_JOINS",
    "TAG_SEMANTIC_KIND",
    "TAG_SEMANTIC_PK",
    "TAG_SENSITIVITY",
    "TAG_TABLE_KEY",
    "get_semantic_view_tags",
    "semantic_view",
]
