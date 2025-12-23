"""Tests for export format registry and capability classification."""

from __future__ import annotations

from codeintel.serving.export.formats import (
    EXPORT_FORMATS,
    default_export_format,
    export_format_choices,
    is_binary_export_format,
    is_text_export_format,
    supports_byte_chunks,
    supports_line_chunks,
    supports_preview,
)
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_export_format_choices_are_supported_and_stable() -> None:
    """Ensure export format choices are stable and registered."""
    formats = export_format_choices()
    expect_true(formats, message="Expected at least one export format")
    expect_equal(tuple(dict.fromkeys(formats)), formats)  # no duplicates
    for fmt in formats:
        expect_true(fmt in EXPORT_FORMATS, message=f"Expected {fmt} to be registered")


def test_default_export_format_is_registered() -> None:
    """Ensure the default export format is part of the registry."""
    fmt = default_export_format()
    expect_true(fmt in EXPORT_FORMATS, message="Expected default export format to be registered")


def test_export_format_capability_classification() -> None:
    """Validate capability helpers for each export format."""
    for fmt in export_format_choices():
        if fmt in {"json", "jsonl"}:
            expect_true(is_text_export_format(fmt))
            expect_true(supports_preview(fmt))
            expect_true(not is_binary_export_format(fmt))
        else:
            expect_true(not is_text_export_format(fmt))
            expect_true(not supports_preview(fmt))
            expect_true(is_binary_export_format(fmt))

        if fmt == "jsonl":
            expect_true(supports_line_chunks(fmt))
        else:
            expect_true(not supports_line_chunks(fmt))

        if fmt in {"parquet", "arrow"}:
            expect_true(supports_byte_chunks(fmt))
        else:
            expect_true(not supports_byte_chunks(fmt))
