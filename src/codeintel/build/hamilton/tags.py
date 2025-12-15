"""Canonical Hamilton tag keys and node type values."""

from __future__ import annotations

TAG_DOMAIN = "domain"
TAG_TARGET = "target"
TAG_TABLE_KEY = "table_key"
TAG_ARTIFACT = "artifact"
TAG_NODE_TYPE = "node_type"

NODE_TYPE_LOADER_QUERY = "loader.query"
NODE_TYPE_LOADER_DATAFRAME = "loader.dataframe"
NODE_TYPE_DATASET = "dataset"
NODE_TYPE_COMPUTE = "compute"
NODE_TYPE_MATERIALIZE = "materialize"
NODE_TYPE_ARTIFACT = "artifact"
NODE_TYPE_TOOL = "tool"

__all__ = [
    "NODE_TYPE_ARTIFACT",
    "NODE_TYPE_COMPUTE",
    "NODE_TYPE_DATASET",
    "NODE_TYPE_LOADER_DATAFRAME",
    "NODE_TYPE_LOADER_QUERY",
    "NODE_TYPE_MATERIALIZE",
    "NODE_TYPE_TOOL",
    "TAG_ARTIFACT",
    "TAG_DOMAIN",
    "TAG_NODE_TYPE",
    "TAG_TABLE_KEY",
    "TAG_TARGET",
]
