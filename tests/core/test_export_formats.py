"""Tests for export format alias normalization."""

from __future__ import annotations

from codeintel.core.exports.formats import (
    mime_type_for_export_format,
    normalize_export_format,
    suffix_for_export_format,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_normalize_ndjson_alias() -> None:
    """Verify ndjson normalizes to the canonical jsonl format."""
    expect_equal(normalize_export_format("ndjson"), "jsonl")


def test_ndjson_suffix_and_mime_type() -> None:
    """Verify ndjson keeps a stable suffix and mime type."""
    expect_equal(suffix_for_export_format("ndjson"), ".ndjson")
    expect_equal(suffix_for_export_format("jsonl"), ".jsonl")
    expect_equal(mime_type_for_export_format("ndjson"), "application/x-ndjson")
