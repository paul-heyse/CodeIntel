"""MCP prompt templates for guided workflows.

Prompts are discoverable via MCP protocol's `list_prompts()` method.
LLM clients can request them to get guided workflows for common tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast
from weakref import WeakKeyDictionary

from fastmcp import Context, FastMCP
from fastmcp.prompts import Message
from mcp import McpError
from mcp.types import PromptMessage

from codeintel.serving.errors import CodeIntelDomainError
from codeintel.serving.export.formats import default_export_format, export_format_choices
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type, parse_filter_value
from codeintel.serving.semantic.models import Op
from codeintel.serving.settings import ServingSettings
from codeintel.serving.uris import META_VIEWS_SQL_DIFF_URI, META_VIEWS_SQL_URI


@dataclass(frozen=True, slots=True)
class _FilterDraft:
    column: str
    op: str
    value: object


@dataclass(slots=True)
class PromptRegistry:
    """Registry of prompt names registered on a FastMCP instance."""

    names: set[str] = field(default_factory=set)

    def register(self, name: str) -> None:
        self.names.add(name)


_PROMPT_REGISTRY: WeakKeyDictionary[FastMCP, PromptRegistry] = WeakKeyDictionary()


def _prompt_registry(mcp: FastMCP) -> PromptRegistry:
    registry = _PROMPT_REGISTRY.get(mcp)
    if registry is None:
        registry = PromptRegistry()
        _PROMPT_REGISTRY[mcp] = registry
    return registry


def list_prompt_names(mcp: FastMCP) -> set[str]:
    """Return registered prompt names for the given MCP server.

    Returns
    -------
    set[str]
        Prompt names registered on the server.
    """
    return set(_prompt_registry(mcp).names)


def register_prompts(
    mcp: FastMCP,
    *,
    settings: ServingSettings,
    kernel: ServingOperations | None = None,
) -> None:
    """Register guided prompts for common workflows.

    Parameters
    ----------
    mcp
        FastMCP application to register prompts on.
    settings
        Serving settings (controls optional capability guidance).
    kernel
        Optional semantic query kernel for schema-aware prompt guidance.
    """
    _register_explore_prompt(mcp)
    _register_export_wizard(mcp, settings=settings)
    _register_query_wizard(mcp, settings=settings, kernel=kernel)
    _register_snapshot_diff_prompt(mcp)


def _tool_invocation(tool: str, /, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "arguments": {k: v for k, v in arguments.items() if v is not None}}


def _tool_invocation_json(tool: str, /, **arguments: object) -> str:
    return json.dumps(_tool_invocation(tool, **arguments), indent=2, sort_keys=True, default=str)


def _register_explore_prompt(mcp: FastMCP) -> None:
    _prompt_registry(mcp).register("explore_codebase")

    @mcp.prompt(
        name="explore_codebase",
        description="Workflow: discover views, schemas, and query examples.",
        tags={"onboarding", "semantic"},
        meta={"version": 2},
    )
    def explore_codebase() -> list[PromptMessage]:
        return [
            Message(
                "Call `semantic_catalog()` to list available views, then choose a view and call "
                "`semantic_describe(view_id=...)`.",
                role="assistant",
            ),
            Message(
                (
                    "Use `semantic_query(request=...)` "
                    "for a small preview. "
                    "If the result is truncated, use `semantic_export(request=...)`."
                ),
                role="assistant",
            ),
            Message(
                (
                    "Use `code_search(query=..., kinds=...)` to locate symbols/files "
                    "when you have a name or pattern."
                ),
                role="assistant",
            ),
        ]


def _register_export_wizard(mcp: FastMCP, *, settings: ServingSettings) -> None:
    _prompt_registry(mcp).register("wizard_export_data")

    @mcp.prompt(
        name="wizard_export_data",
        description="Interactive export wizard (uses elicitation when supported).",
        tags={"export", "wizard"},
        meta={"version": 1},
    )
    async def wizard_export_data(ctx: Context) -> list[PromptMessage]:
        no_elicitation = _wizard_export_data_no_elicitation(settings)
        cancelled = [Message("Export wizard cancelled.", role="assistant")]
        messages: list[PromptMessage] | None = None

        if not _supports_elicitation(ctx):
            messages = no_elicitation
        else:
            format_choices = [str(fmt) for fmt in export_format_choices()]
            accepted_view = await _try_elicit(
                ctx,
                "Which view_id do you want to export?",
                response_type=str,
            )
            if accepted_view is None:
                messages = no_elicitation
            else:
                view_result = _accepted_data(accepted_view)
                view_id = str(view_result).strip() if view_result is not None else ""
                if not view_id:
                    messages = cancelled if view_result is None else no_elicitation
                else:
                    accepted_format = await _try_elicit(
                        ctx,
                        "Choose an export format.",
                        response_type=format_choices,
                    )
                    if accepted_format is None:
                        messages = no_elicitation
                    else:
                        format_result = _accepted_data(accepted_format)
                        if format_result is None:
                            messages = cancelled
                        else:
                            export_format = str(format_result).strip() or default_export_format()
                            messages = [
                                Message(
                                    (
                                        "Call `semantic_export` using the parameters below "
                                        "(task-capable when supported):"
                                    ),
                                    role="assistant",
                                ),
                                Message(
                                    _tool_invocation_json(
                                        "semantic_export",
                                        request={
                                            "view_id": view_id,
                                            "export_format": export_format,
                                            "limit": 100_000,
                                        },
                                    ),
                                    role="assistant",
                                ),
                                Message(
                                    (
                                        "Then call `resources/read` on `meta_uri` "
                                        "to discover safe retrieval URIs, "
                                        "and fetch payload in chunks when needed."
                                    ),
                                    role="assistant",
                                ),
                            ]

        return messages if messages is not None else no_elicitation


def _register_query_wizard(
    mcp: FastMCP, *, settings: ServingSettings, kernel: ServingOperations | None
) -> None:
    feature_set = ServingFeatureSet.from_settings(settings)
    _prompt_registry(mcp).register("wizard_query_view")

    @mcp.prompt(
        name="wizard_query_view",
        description="Interactive query wizard (uses elicitation when supported).",
        tags={"semantic", "wizard"},
        meta={"version": 1},
    )
    async def wizard_query_view(ctx: Context) -> list[PromptMessage]:
        if not _supports_elicitation(ctx):
            return _wizard_query_view_no_elicitation(settings)

        accepted_view = await _try_elicit(
            ctx,
            "Which view_id do you want to query?",
            response_type=str,
        )
        if accepted_view is None:
            return _wizard_query_view_no_elicitation(settings)
        view_result = _accepted_data(accepted_view)
        if view_result is None:
            return [Message("Query wizard cancelled.", role="assistant")]
        view_id = str(view_result).strip()
        if not view_id:
            return _wizard_query_view_no_elicitation(settings)

        columns = _maybe_get_columns(kernel, view_id=view_id)
        select = await _maybe_elicit_select(ctx, columns=columns)
        filter_draft = await _maybe_elicit_filter(
            ctx,
            columns=columns,
            kernel=kernel,
            view_id=view_id,
        )

        filters: list[dict[str, object]] | None = None
        if filter_draft is not None:
            filters = [
                {
                    "column": filter_draft.column,
                    "op": filter_draft.op,
                    "value": filter_draft.value,
                }
            ]

        return [
            Message("Start with a schema check:", role="assistant"),
            Message(_tool_invocation_json("semantic_describe", view_id=view_id), role="assistant"),
            Message(
                "Then run a small preview query (add filters/select/order_by as needed):",
                role="assistant",
            ),
            Message(
                _tool_invocation_json(
                    "semantic_query",
                    request={
                        "view_id": view_id,
                        "filters": filters,
                        "select": select,
                        "pagination": {"limit": 10, "offset": 0},
                    },
                ),
                role="assistant",
            ),
            Message(
                (
                    "Need Arrow IPC? Add `export_format: \"arrow\"` to semantic_query "
                    "to receive an export handle."
                ),
                role="assistant",
            ),
            Message(
                (
                    "If sampling is enabled server-side and supported client-side, "
                    "large results may include a summary. "
                    f"sampling_enabled={feature_set.enable_mcp_sampling}"
                ),
                role="assistant",
            ),
        ]


def _register_snapshot_diff_prompt(mcp: FastMCP) -> None:
    _prompt_registry(mcp).register("what_changed_between_snapshots")

    @mcp.prompt(
        name="what_changed_between_snapshots",
        description="Workflow: review semantic view SQL diffs between snapshots.",
        tags={"ops", "meta"},
        meta={"version": 1},
    )
    def what_changed_between_snapshots() -> list[PromptMessage]:
        return [
            Message(
                f"Call `resources/read` on `{META_VIEWS_SQL_DIFF_URI}` (if present).",
                role="assistant",
            ),
            Message(
                (
                    f"Then use `resources/read` on `{META_VIEWS_SQL_URI}` "
                    "for full compiled SQL if needed."
                ),
                role="assistant",
            ),
        ]


def _supports_elicitation(ctx: Context) -> bool:
    try:
        _ = ctx.session
    except RuntimeError:
        return False
    return True


async def _try_elicit(
    ctx: Context,
    message: str,
    *,
    response_type: type[object] | list[str] | None,
) -> object | None:
    try:
        if response_type is None:
            return await ctx.elicit(message, response_type=None)
        if isinstance(response_type, list):
            return await ctx.elicit(message, response_type=response_type)
        return await ctx.elicit(message, response_type=response_type)
    except (McpError, RuntimeError):
        return None


def _accepted_data(result: object | None) -> object | None:
    if result is None:
        return None
    action = getattr(result, "action", None)
    if action != "accept":
        return None
    return getattr(result, "data", None)


def _wizard_export_data_no_elicitation(settings: ServingSettings) -> list[PromptMessage]:
    feature_set = ServingFeatureSet.from_settings(settings)
    return [
        Message("Elicitation is not available in this client.", role="assistant"),
        Message(
            "Use `semantic_catalog` to choose a view_id, then call `semantic_export(request=...)`.",
            role="assistant",
        ),
        Message(
            (
                "Export formats: jsonl/json (text) and parquet/arrow (binary). "
                f"export_tasks_enabled={feature_set.enable_mcp_export_tasks}"
            ),
            role="assistant",
        ),
    ]


def _wizard_query_view_no_elicitation(settings: ServingSettings) -> list[PromptMessage]:
    feature_set = ServingFeatureSet.from_settings(settings)
    return [
        Message("Elicitation is not available in this client.", role="assistant"),
        Message(
            (
                "Call `semantic_describe(view_id=...)` then `semantic_query(request=...)`. "
                "Use `export_format: \"arrow\"` to get IPC exports."
            ),
            role="assistant",
        ),
        Message(f"sampling_enabled={feature_set.enable_mcp_sampling}", role="assistant"),
    ]


def _maybe_get_columns(kernel: ServingOperations | None, *, view_id: str) -> list[str] | None:
    if kernel is None:
        return None
    try:
        desc = kernel.describe(view_id)
    except (CodeIntelDomainError, KeyError, TypeError, ValueError):
        return None
    columns = [col for col in desc.columns if isinstance(col, str) and col]
    return columns or None


async def _maybe_elicit_select(ctx: Context, *, columns: list[str] | None) -> list[str] | None:
    if columns is None:
        return None
    prompt = (
        "Enter a comma-separated list of columns to select, or cancel/leave blank "
        "to select all columns. "
        f"Available columns: {', '.join(columns[:25])}"
    )
    accepted = await _try_elicit(ctx, prompt, response_type=str)
    data = _accepted_data(accepted)
    if data is None:
        return None
    raw = str(data).strip()
    if not raw:
        return None
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = set(columns)
    filtered = [item for item in requested if item in allowed]
    return filtered or None


async def _maybe_elicit_filter(
    ctx: Context,
    *,
    columns: list[str] | None,
    kernel: ServingOperations | None,
    view_id: str,
) -> _FilterDraft | None:
    if columns is None:
        return None
    accepted_add = await _try_elicit(ctx, "Add a filter?", response_type=["no", "yes"])
    add_data = _accepted_data(accepted_add)
    if add_data is None or str(add_data) != "yes":
        return None

    accepted_column = await _try_elicit(ctx, "Choose a column to filter on.", response_type=columns)
    column_data = _accepted_data(accepted_column)
    column = str(column_data).strip() if column_data is not None else ""
    if not column:
        return None

    column_types = _maybe_get_column_types(kernel, view_id=view_id)
    dtype = column_types.get(column) if column_types is not None else None
    ops: list[str] = [str(item) for item in allowed_ops_for_column_type(dtype)]

    accepted_op = await _try_elicit(ctx, "Choose an operator.", response_type=ops)
    op_data = _accepted_data(accepted_op)
    op = str(op_data).strip() if op_data is not None else ""
    if not op or op not in ops:
        return None

    accepted_value = await _try_elicit(ctx, "Enter a value.", response_type=str)
    value_data = _accepted_data(accepted_value)
    raw = str(value_data).strip() if value_data is not None else ""
    if not raw:
        return None

    return _FilterDraft(
        column=column, op=op, value=parse_filter_value(dtype, op=cast("Op", op), raw=raw)
    )


def _maybe_get_column_types(
    kernel: ServingOperations | None,
    *,
    view_id: str,
) -> dict[str, str] | None:
    if kernel is None:
        return None
    try:
        desc = kernel.describe(view_id)
    except (CodeIntelDomainError, KeyError, TypeError, ValueError):
        return None
    return {str(k): str(v) for k, v in desc.column_types.items()}


__all__ = ["list_prompt_names", "register_prompts"]
