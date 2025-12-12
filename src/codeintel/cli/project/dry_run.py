"""Dry-run execution planning for CLI operations.

Provide functions to plan operation execution without actually running them,
enabling users to preview what would happen.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from codeintel.cli.core.result_types import DryRunResult, DryRunStep
from codeintel.cli.rendering.types import OutputFormat
from codeintel.serving.operations.catalog import get_operation


def plan_dry_run(
    op_id: str,
    params: dict[str, Any],
    *,
    skip_prereqs: bool = False,
) -> DryRunResult:
    """Plan a dry-run execution of an operation.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Operation parameters.
    skip_prereqs
        Whether prerequisites would be skipped.

    Returns
    -------
    DryRunResult
        Execution plan without actual execution.
    """
    operation = get_operation(op_id)
    if operation is None:
        return DryRunResult(
            target_operation=op_id,
            steps=[],
            warnings=[f"Unknown operation: {op_id}"],
        )

    steps: list[DryRunStep] = []
    warnings: list[str] = []

    if not skip_prereqs:
        steps.append(
            DryRunStep(
                operation_id="prerequisites",
                description="Run prerequisite pipeline for data dependencies",
                params={},
                is_prereq=True,
            )
        )
        warnings.append(
            "Prerequisites will be checked and run if needed. Use --skip-prereqs to skip."
        )

    steps.append(
        DryRunStep(
            operation_id=op_id,
            description=operation.summary or op_id,
            params=params,
            is_prereq=False,
        )
    )

    return DryRunResult(
        target_operation=op_id,
        steps=steps,
        estimated_duration=None,
        warnings=warnings,
    )


def render_dry_run(plan: DryRunResult, output_format: OutputFormat) -> None:
    """Render dry-run plan to stdout.

    Parameters
    ----------
    plan
        Execution plan.
    output_format
        Output format.
    """
    render_dry_run_to(plan, output_format, sys.stdout)


def render_dry_run_to(
    plan: DryRunResult,
    output_format: OutputFormat,
    writer: TextIO,
) -> None:
    """Render dry-run plan to the specified writer.

    Parameters
    ----------
    plan
        Execution plan.
    output_format
        Output format.
    writer
        Text writer for output.
    """
    if output_format == OutputFormat.JSON:
        writer.write(json.dumps(plan.to_dict(), indent=2))
        writer.write("\n")
        return

    _render_dry_run_text(plan, writer)


def _render_dry_run_text(plan: DryRunResult, writer: TextIO) -> None:
    """Render dry-run plan as human-readable text.

    Parameters
    ----------
    plan
        Execution plan.
    writer
        Text writer for output.
    """
    writer.write(f"Dry-run plan for: {plan.target_operation}\n")
    writer.write("-" * 50 + "\n")

    if not plan.steps:
        writer.write("No steps planned.\n")
    else:
        for i, step in enumerate(plan.steps, 1):
            prefix = "[prereq]" if step.is_prereq else "[target]"
            writer.write(f"{i}. {prefix} {step.operation_id}\n")
            writer.write(f"   {step.description}\n")
            if step.params:
                params_str = ", ".join(f"{k}={v}" for k, v in step.params.items())
                writer.write(f"   Params: {params_str}\n")

    if plan.estimated_duration:
        writer.write(f"\nEstimated duration: {plan.estimated_duration}\n")

    if plan.warnings:
        writer.write("\nNotes:\n")
        for warning in plan.warnings:
            writer.write(f"  - {warning}\n")


__all__ = [
    "plan_dry_run",
    "render_dry_run",
    "render_dry_run_to",
]
