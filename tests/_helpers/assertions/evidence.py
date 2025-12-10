"""Evidence assertion helpers for analytics tests."""

from __future__ import annotations

from collections.abc import Mapping

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_in


def assert_evidence_snippet_contains(sample: Mapping[str, object], substring: str) -> None:
    """Assert evidence snippet contains the expected substring.

    Raises
    ------
    TypeError
        If the snippet is missing or not a string.
    """
    snippet = sample.get("snippet")
    if not isinstance(snippet, str):
        message = "evidence snippet missing or not a string"
        raise TypeError(message)
    expect_in(substring, snippet)


def assert_evidence_location(
    sample: Mapping[str, object],
    *,
    path: str,
    lineno: int | None = None,
) -> None:
    """Assert evidence path and optional line number."""
    expect_equal(sample.get("path"), path)
    if lineno is not None:
        expect_equal(sample.get("lineno"), lineno)


def assert_evidence_urn(sample: Mapping[str, object], expected_urn: str) -> None:
    """Assert evidence carries the expected URN field."""
    expect_equal(sample.get("urn"), expected_urn)


__all__ = [
    "assert_evidence_location",
    "assert_evidence_snippet_contains",
    "assert_evidence_urn",
]
