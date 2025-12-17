"""MCP prompt templates for guided workflows.

Prompts are discoverable via MCP protocol's `list_prompts()` method.
LLM clients can request them to get guided workflows for common tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.mcp._compat import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Register guided prompts for common workflows.

    Parameters
    ----------
    mcp
        FastMCP application to register prompts on.
    """

    @mcp.prompt()
    def explore_codebase() -> str:
        """Guided workflow for exploring an unfamiliar codebase.

        Returns
        -------
        str
            Step-by-step instructions for codebase exploration.
        """
        return """To explore this codebase:
1. Call semantic_catalog() to see available data views
2. Pick a view and call semantic_describe(view_id=...) for its schema
3. Use semantic_query(view_id=...) to fetch sample data
4. Use code_search(query=...) to find specific code patterns"""

    @mcp.prompt()
    def find_function(name: str) -> str:
        """Guided workflow for finding and understanding a function.

        Parameters
        ----------
        name
            Name of the function to find.

        Returns
        -------
        str
            Step-by-step instructions for locating the function.
        """
        return f"""To find function '{name}':
1. Use code_search(query="{name}") to locate it
2. Get function details via semantic_query(view_id="analytics.function_metrics",
   filters=[{{"column": "name", "op": "contains", "value": "{name}"}}])
3. Check callers via semantic_query(view_id="graph.call_edges",
   filters=[{{"column": "callee_name", "op": "eq", "value": "{name}"}}])"""

    @mcp.prompt()
    def export_data(view_id: str) -> str:
        """Guided workflow for exporting large datasets.

        Parameters
        ----------
        view_id
            The semantic view to export.

        Returns
        -------
        str
            Step-by-step instructions for data export.
        """
        return f"""To export data from '{view_id}':
1. Preview with semantic_query(view_id="{view_id}", limit=10)
2. For full export, call semantic_export(view_id="{view_id}", format="ndjson")
3. The response includes a resource URI - fetch it to download the data
4. NDJSON format is recommended for large datasets (streaming-friendly)"""

    @mcp.prompt()
    def analyze_metrics() -> str:
        """Guided workflow for analyzing code quality metrics.

        Returns
        -------
        str
            Step-by-step instructions for code quality analysis.
        """
        return """To analyze code quality:
1. List metrics views: semantic_catalog() (look for 'analytics.*' views)
2. Describe metrics: semantic_describe(view_id="analytics.function_metrics")
3. Find complex functions: semantic_query(view_id="analytics.function_metrics",
   filters=[{"column": "cyclomatic_complexity", "op": "gt", "value": 10}],
   order_by=["-cyclomatic_complexity"])
4. Use semantic_explain() to see the underlying SQL if needed"""

    @mcp.prompt()
    def get_server_status() -> str:
        """Guided workflow for checking server status.

        Returns
        -------
        str
            Step-by-step instructions for checking server status.
        """
        return """To check CodeIntel server status:
1. Call serving_meta() to get snapshot and version information
2. The response includes repo, commit, and run_id for provenance tracking
3. All queries reference this snapshot - data is consistent within a session"""


__all__ = ["register_prompts"]
