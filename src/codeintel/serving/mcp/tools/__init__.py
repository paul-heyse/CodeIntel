"""FastMCP tool registration modules for CodeIntel serving."""

from codeintel.serving.mcp.tools.catalog import register_catalog_tool
from codeintel.serving.mcp.tools.describe import register_describe_tool
from codeintel.serving.mcp.tools.explain import register_explain_tool
from codeintel.serving.mcp.tools.export import register_export_tool
from codeintel.serving.mcp.tools.meta import register_meta_tool
from codeintel.serving.mcp.tools.query import register_query_tool
from codeintel.serving.mcp.tools.search import register_search_tool

__all__ = [
    "register_catalog_tool",
    "register_describe_tool",
    "register_explain_tool",
    "register_export_tool",
    "register_meta_tool",
    "register_query_tool",
    "register_search_tool",
]
