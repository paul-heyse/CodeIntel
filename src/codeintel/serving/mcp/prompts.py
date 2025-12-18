"""MCP prompt templates for guided workflows.

Prompts are discoverable via MCP protocol's `list_prompts()` method.
LLM clients can request them to get guided workflows for common tasks.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastmcp.prompts import Message
from fastmcp.server.dependencies import CurrentContext

from codeintel.serving.mcp._compat import Context

if TYPE_CHECKING:
    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.settings import ServingSettings

CURRENT_CONTEXT: object = CurrentContext()


def register_prompts(mcp: FastMCP, *, settings: ServingSettings) -> None:
    """Register guided prompts for common workflows.

    Parameters
    ----------
    mcp
        FastMCP application to register prompts on.
    settings
        Serving settings (controls optional capability guidance).
    """

    @mcp.prompt(
        name="explore_codebase",
        description="Workflow: discover views, schemas, and query examples.",
        tags={"onboarding", "semantic"},
        meta={"version": 2},
    )
    def explore_codebase() -> list[object]:
        return [
            Message(
                "Call `semantic_catalog()` to list available views, then choose a view and call "
                "`semantic_describe(view_id=...)`.",
                role="assistant",
            ),
            Message(
                "Use `semantic_query(view_id=..., filters=..., pagination=...)` for a small preview. "
                "If the result is truncated, use `semantic_export(...)`.",
                role="assistant",
            ),
            Message(
                "Use `code_search(query=..., kinds=...)` to locate symbols/files when you have a name or pattern.",
                role="assistant",
            ),
        ]

    @mcp.prompt(
        name="wizard_export_data",
        description="Interactive export wizard (uses elicitation when supported).",
        tags={"export", "wizard"},
        meta={"version": 1},
    )
    async def wizard_export_data(ctx: Context = CURRENT_CONTEXT) -> list[object]:
        format_choices = ["ndjson", "json", "parquet", "arrow"]
        accepted_view = await ctx.elicit("Which view_id do you want to export?", response_type=str)
        if getattr(accepted_view, "action", None) != "accept":
            return [Message("Export wizard cancelled.", role="assistant")]
        view_id = str(getattr(accepted_view, "data", "") or "").strip()

        accepted_format = await ctx.elicit(
            "Choose an export format.",
            response_type=format_choices,
        )
        if getattr(accepted_format, "action", None) != "accept":
            return [Message("Export wizard cancelled.", role="assistant")]
        export_format = str(getattr(accepted_format, "data", "ndjson"))

        return [
            Message(
                "Call `semantic_export` using the parameters below (task-capable when supported):",
                role="assistant",
            ),
            Message(
                _tool_invocation_json(
                    "semantic_export",
                    view_id=view_id,
                    export_format=export_format,
                    limit=100_000,
                ),
                role="assistant",
            ),
            Message(
                "Then call `resources/read` on `meta_uri` to discover safe retrieval URIs, and fetch payload in chunks "
                "when needed.",
                role="assistant",
            ),
        ]

    @mcp.prompt(
        name="wizard_query_view",
        description="Interactive query wizard (uses elicitation when supported).",
        tags={"semantic", "wizard"},
        meta={"version": 1},
    )
    async def wizard_query_view(ctx: Context = CURRENT_CONTEXT) -> list[object]:
        accepted_view = await ctx.elicit("Which view_id do you want to query?", response_type=str)
        if getattr(accepted_view, "action", None) != "accept":
            return [Message("Query wizard cancelled.", role="assistant")]
        view_id = str(getattr(accepted_view, "data", "") or "").strip()

        return [
            Message("Start with a schema check:", role="assistant"),
            Message(_tool_invocation_json("semantic_describe", view_id=view_id), role="assistant"),
            Message(
                "Then run a small preview query (add filters/select/order_by as needed):",
                role="assistant",
            ),
            Message(
                _tool_invocation_json("semantic_query", view_id=view_id, pagination={"limit": 10, "offset": 0}),
                role="assistant",
            ),
            Message(
                (
                    "If sampling is enabled server-side and supported client-side, large results may include a summary. "
                    f"sampling_enabled={settings.mcp_enable_sampling}"
                ),
                role="assistant",
            ),
        ]

    @mcp.prompt(
        name="what_changed_between_snapshots",
        description="Workflow: review semantic view SQL diffs between snapshots.",
        tags={"ops", "meta"},
        meta={"version": 1},
    )
    def what_changed_between_snapshots() -> list[object]:
        return [
            Message("Call `resources/read` on `codeintel://meta/views_sql_diff` (if present).", role="assistant"),
            Message("Then use `resources/read` on `codeintel://meta/views_sql` for full compiled SQL if needed.", role="assistant"),
        ]


def _tool_invocation(tool: str, /, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "arguments": {k: v for k, v in arguments.items() if v is not None}}


def _tool_invocation_json(tool: str, /, **arguments: object) -> str:
    return json.dumps(_tool_invocation(tool, **arguments), indent=2, sort_keys=True, default=str)


__all__ = ["register_prompts"]
