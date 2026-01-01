"""Normalization helpers for DuckDB explain outputs."""

from __future__ import annotations


def normalize_explain_output(plan: str | None) -> str | None:
    """Normalize explain text for deterministic comparisons.

    Parameters
    ----------
    plan
        Explain output text from DuckDB.

    Returns
    -------
    str | None
        Normalized explain text with stable whitespace handling.
    """
    if plan is None:
        return None
    lines = [line.rstrip() for line in plan.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


__all__ = ["normalize_explain_output"]
