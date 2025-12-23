"""Canonical Hamilton tag keys and node type values.

This module is intentionally dependency-light and shared across layers.
"""

from __future__ import annotations

TAG_DOMAIN = "domain"
TAG_TARGET = "target"
TAG_TABLE_KEY = "table_key"
TAG_ARTIFACT = "artifact"
TAG_ARTIFACT_PATH_TEMPLATE = "artifact_path_template"
TAG_NODE_TYPE = "node_type"
TAG_TARGET_RESOURCES = "target_resources"
TAG_TARGET_EXECUTION = "target_execution"
TAG_TARGET_PARAMETERS = "target_parameters"
TAG_TARGET_ESTIMATED_DURATION_MS = "target_estimated_duration_ms"
TAG_TARGET_SPEC_VERSION = "target_spec_version"

TAG_OUTPUT_KIND = "output_kind"
TAG_SEMANTIC_ID = "semantic_id"
TAG_ENTITY = "entity"
TAG_GRAIN = "grain"
TAG_MCP_VISIBLE = "mcp_visible"

OUTPUT_KIND_VIEW = "view"
OUTPUT_KIND_SEMANTIC_VIEW = "semantic_view"

NODE_TYPE_LOADER_QUERY = "loader.query"
NODE_TYPE_LOADER_DATAFRAME = "loader.dataframe"
NODE_TYPE_DATASET = "dataset"
NODE_TYPE_COMPUTE = "compute"
NODE_TYPE_MATERIALIZE = "materialize"
NODE_TYPE_ARTIFACT = "artifact"
NODE_TYPE_TOOL = "tool"
NODE_TYPE_HELPER = "helper"

__all__ = [
    "NODE_TYPE_ARTIFACT",
    "NODE_TYPE_COMPUTE",
    "NODE_TYPE_DATASET",
    "NODE_TYPE_HELPER",
    "NODE_TYPE_LOADER_DATAFRAME",
    "NODE_TYPE_LOADER_QUERY",
    "NODE_TYPE_MATERIALIZE",
    "NODE_TYPE_TOOL",
    "OUTPUT_KIND_SEMANTIC_VIEW",
    "OUTPUT_KIND_VIEW",
    "TAG_ARTIFACT",
    "TAG_ARTIFACT_PATH_TEMPLATE",
    "TAG_DOMAIN",
    "TAG_ENTITY",
    "TAG_GRAIN",
    "TAG_MCP_VISIBLE",
    "TAG_NODE_TYPE",
    "TAG_OUTPUT_KIND",
    "TAG_SEMANTIC_ID",
    "TAG_TABLE_KEY",
    "TAG_TARGET",
    "TAG_TARGET_ESTIMATED_DURATION_MS",
    "TAG_TARGET_EXECUTION",
    "TAG_TARGET_PARAMETERS",
    "TAG_TARGET_RESOURCES",
    "TAG_TARGET_SPEC_VERSION",
]
