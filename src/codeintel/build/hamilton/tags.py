"""Backward compatibility alias for Hamilton tag constants.

The canonical location for these constants is `codeintel.hamilton.tags`.
"""

from __future__ import annotations

from codeintel.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_COMPUTE,
    NODE_TYPE_DATASET,
    NODE_TYPE_LOADER_DATAFRAME,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
    NODE_TYPE_TOOL,
    OUTPUT_KIND_SEMANTIC_VIEW,
    OUTPUT_KIND_VIEW,
    TAG_ARTIFACT,
    TAG_DOMAIN,
    TAG_ENTITY,
    TAG_GRAIN,
    TAG_MCP_VISIBLE,
    TAG_NODE_TYPE,
    TAG_OUTPUT_KIND,
    TAG_SEMANTIC_ID,
    TAG_TABLE_KEY,
    TAG_TARGET,
)

__all__ = [
    "NODE_TYPE_ARTIFACT",
    "NODE_TYPE_COMPUTE",
    "NODE_TYPE_DATASET",
    "NODE_TYPE_LOADER_DATAFRAME",
    "NODE_TYPE_LOADER_QUERY",
    "NODE_TYPE_MATERIALIZE",
    "NODE_TYPE_TOOL",
    "OUTPUT_KIND_SEMANTIC_VIEW",
    "OUTPUT_KIND_VIEW",
    "TAG_ARTIFACT",
    "TAG_DOMAIN",
    "TAG_ENTITY",
    "TAG_GRAIN",
    "TAG_MCP_VISIBLE",
    "TAG_NODE_TYPE",
    "TAG_OUTPUT_KIND",
    "TAG_SEMANTIC_ID",
    "TAG_TABLE_KEY",
    "TAG_TARGET",
]
