"""Tests for export format normalization."""

from __future__ import annotations

import pytest

from codeintel.core.exports.formats import (
    mime_type_for_export_format,
    normalize_export_format,
    suffix_for_export_format,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_normalize_rejects_ndjson() -> None:
    """Verify ndjson is rejected as an unsupported alias."""
    with pytest.raises(ValueError, match="Unsupported export format"):
        normalize_export_format("ndjson")


def test_jsonl_suffix_and_mime_type() -> None:
    """Verify jsonl keeps a stable suffix and mime type."""
    expect_equal(suffix_for_export_format("jsonl"), ".jsonl")
    expect_equal(mime_type_for_export_format("jsonl"), "application/x-ndjson")
