"""Preflight checks for planning and execution readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codeintel.build.hamilton.optional_inputs import optional_inputs_for_target

PreflightIssueKind = Literal["missing_input_table"]
PreflightSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """Preflight issue describing an unmet prerequisite."""

    kind: PreflightIssueKind
    target: str | None
    message: str
    severity: PreflightSeverity

    def to_block_reason(self) -> str:
        """Return a stable reason string for block mapping.

        Returns
        -------
        str
            Block reason string for planning outputs.
        """
        return f"{self.kind}:{self.message}"


def missing_input_issues(
    *,
    missing_required: tuple[str, ...],
    missing_optional: tuple[str, ...],
    target: str,
) -> tuple[PreflightIssue, ...]:
    """Build missing-input issues for a target.

    Returns
    -------
    tuple[PreflightIssue, ...]
        Issues describing missing input tables.
    """
    issues: list[PreflightIssue] = []
    if missing_required:
        issues.append(
            PreflightIssue(
                kind="missing_input_table",
                target=target,
                message=f"Missing input tables: {', '.join(missing_required)}",
                severity="error",
            )
        )
    if missing_optional:
        issues.append(
            PreflightIssue(
                kind="missing_input_table",
                target=target,
                message=f"Missing optional input tables: {', '.join(missing_optional)}",
                severity="warning",
            )
        )
    return tuple(issues)


def classify_missing_inputs(
    *,
    optional_inputs: frozenset[str],
    missing: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split missing inputs into required and optional buckets.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Missing required and missing optional table keys.
    """
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for table_key in sorted(missing):
        if table_key in optional_inputs:
            missing_optional.append(table_key)
        else:
            missing_required.append(table_key)
    return tuple(missing_required), tuple(missing_optional)


def optional_inputs_for_targets(target: str) -> frozenset[str]:
    """Return optional input table keys for a target.

    Returns
    -------
    frozenset[str]
        Optional input table keys for the target.
    """
    return optional_inputs_for_target(target)


__all__ = [
    "PreflightIssue",
    "PreflightIssueKind",
    "PreflightSeverity",
    "classify_missing_inputs",
    "missing_input_issues",
    "optional_inputs_for_targets",
]
