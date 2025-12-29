"""Handlers for `codeintel build validate`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.graph_validation import (
    validate_graph,
    validation_result_to_json,
)
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_execution_failed
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle

if TYPE_CHECKING:
    from codeintel.build.hamilton.validate import GraphValidationIssue, GraphValidationResult
    from codeintel.cli.context import CommandContext

_MAX_ERROR_ISSUES: int = 50


def build_validate_handler(ctx: CommandContext) -> CliResult[str]:
    """Validate Hamilton DAG invariants for the configured build graph.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[str]
        Validation report as JSON text on success.
    """
    output_format = ctx.params.get_str("output_format") or "json"

    if output_format != "json":
        return fail_execution_failed(
            "build",
            f"Unsupported validate output format: {output_format}",
            status=400,
        )

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    result = validate_graph(runtime=runtime_bundle)

    if result.has_errors:
        error_summary = _format_error_summary(result)
        return fail_execution_failed("build", error_summary, status=409)

    payload = validation_result_to_json(result, indent=2)
    return CliResult.ok(
        payload,
        metadata={
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        },
    )


def _format_error_summary(result: GraphValidationResult) -> str:
    lines: list[str] = [
        "Hamilton graph validation failed.",
        "",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
        "",
    ]
    lines.extend(_format_issue_line(issue) for issue in result.errors[:_MAX_ERROR_ISSUES])
    if len(result.errors) > _MAX_ERROR_ISSUES:
        lines.append(f"... +{len(result.errors) - _MAX_ERROR_ISSUES} more")
    return "\n".join(lines) + "\n"


def _format_issue_line(issue: GraphValidationIssue) -> str:
    context_parts = _issue_context(issue)
    suffix = f" ({', '.join(context_parts)})" if context_parts else ""
    return f"- {issue.code}: {issue.message}{suffix}"


def _issue_context(issue: GraphValidationIssue) -> list[str]:
    fields = (
        ("node", issue.node),
        ("target", issue.target),
        ("table_key", issue.table_key),
        ("artifact", issue.artifact),
        ("module", issue.module),
        ("origin", issue.origin),
        ("plugin", issue.plugin_name),
    )
    return [f"{label}={value}" for label, value in fields if value]


__all__ = [
    "build_validate_handler",
]
