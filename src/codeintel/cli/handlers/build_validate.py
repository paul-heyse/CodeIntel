"""Handlers for `codeintel build validate`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.graph_validation import (
    validate_graph,
    validation_result_to_json,
)
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_execution_failed

if TYPE_CHECKING:
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

    result = validate_graph()

    if result.has_errors:
        lines: list[str] = []
        lines.append("Hamilton graph validation failed.")
        lines.append("")
        lines.append(f"Errors: {len(result.errors)}")
        lines.append(f"Warnings: {len(result.warnings)}")
        lines.append("")

        for issue in result.errors[:_MAX_ERROR_ISSUES]:
            parts: list[str] = []
            if issue.node:
                parts.append(f"node={issue.node}")
            if issue.target:
                parts.append(f"target={issue.target}")
            if issue.table_key:
                parts.append(f"table_key={issue.table_key}")
            if issue.artifact:
                parts.append(f"artifact={issue.artifact}")
            suffix = f" ({', '.join(parts)})" if parts else ""
            lines.append(f"- {issue.code}: {issue.message}{suffix}")

        if len(result.errors) > _MAX_ERROR_ISSUES:
            lines.append(f"... +{len(result.errors) - _MAX_ERROR_ISSUES} more")

        return fail_execution_failed("build", "\n".join(lines) + "\n", status=409)

    payload = validation_result_to_json(result, indent=2)
    return CliResult.ok(
        payload,
        metadata={
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        },
    )


__all__ = [
    "build_validate_handler",
]
