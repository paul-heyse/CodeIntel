"""Tests for PR-78: Hamilton graph validator gate."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.validate import validate_graph

_MAX_ISSUES: int = 25


def test_pr78_graph_validator_clean_auto() -> None:
    """Verify the auto-mode Hamilton graph satisfies validator invariants."""
    result = validate_graph(mode="auto")
    if result.errors:
        issues = "\n".join(f"- {e.code}: {e.message}" for e in result.errors[:_MAX_ISSUES])
        more = (
            ""
            if len(result.errors) <= _MAX_ISSUES
            else f"\n... +{len(result.errors) - _MAX_ISSUES} more"
        )
        pytest.fail(f"Expected validator to return no errors, got:\n{issues}{more}")
