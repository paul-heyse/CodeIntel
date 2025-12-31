"""Tests for output schema registry filtering."""

from __future__ import annotations

from codeintel.core.schemas.output_registry import (
    NON_INFERABLE_OUTPUT_KEYS,
    OUTPUT_TABLE_SCHEMAS,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_output_registry_filters_non_inferable_outputs() -> None:
    """Output registry should only expose non-inferable schemas."""
    expect_equal(
        set(OUTPUT_TABLE_SCHEMAS),
        expected=set(NON_INFERABLE_OUTPUT_KEYS),
    )
