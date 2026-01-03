"""Tests for validation option defaults."""

from __future__ import annotations

import pytest

from codeintel.core.validation.options import BaseValidationOptions

MAX_FINDINGS_DEFAULT = 5


def test_base_validation_with_defaults_respects_explicit_false() -> None:
    """Explicit False should override default True in with_defaults."""
    defaults = BaseValidationOptions(
        severity_overrides={"*": "error"},
        hard_fail=True,
        max_findings_per_rule=MAX_FINDINGS_DEFAULT,
    )
    overrides = BaseValidationOptions(hard_fail=False)
    merged = overrides.with_defaults(defaults)

    if merged.hard_fail is not False:
        pytest.fail(f"Expected hard_fail False, got {merged.hard_fail}")
    if merged.max_findings_per_rule != MAX_FINDINGS_DEFAULT:
        pytest.fail(
            f"Expected max_findings_per_rule to default to {MAX_FINDINGS_DEFAULT}, "
            f"got {merged.max_findings_per_rule}"
        )
    if merged.severity_overrides != {"*": "error"}:
        pytest.fail(
            "Expected severity_overrides to default to {'*': 'error'}, "
            f"got {merged.severity_overrides}"
        )
