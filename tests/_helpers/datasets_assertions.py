"""Shared assertions for dataset specs and registries."""

from __future__ import annotations

from tests._helpers.assertions import expect_equal, expect_true


def expect_spec_has_columns(spec: object, *, label: str = "schema_columns") -> None:
    """Assert that a dataset spec exposes non-empty schema_columns."""
    columns = getattr(spec, "schema_columns", None)
    expect_true(columns is not None and len(columns) > 0, label=label)


def expect_spec_has_capabilities(spec: object, *, label: str = "capabilities") -> None:
    """Assert that a dataset spec exposes capabilities."""
    capabilities = getattr(spec, "capabilities", None)
    expect_true(capabilities is not None, label=label)


def expect_spec_filename(
    spec: object,
    expected_filename: str | None,
    *,
    label: str = "jsonl_filename",
) -> None:
    """Assert that a dataset spec jsonl_filename matches expectation."""
    filename = getattr(spec, "jsonl_filename", None)
    expect_equal(filename, expected_filename, label=label)
