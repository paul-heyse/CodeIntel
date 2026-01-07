"""Tests for rustworkx normalization helpers."""

from __future__ import annotations

from codeintel.build.graphs.rx import normalize_mapping, stable_key
from tests._helpers.assertions import expect_equal


def test_stable_key_orders_strings() -> None:
    """Stable key ordering should be deterministic for strings."""
    values = ["b", "a"]
    ordered = sorted(values, key=stable_key)
    expect_equal(ordered, ["a", "b"], label="ordered_values")


def test_normalize_mapping_applies_nan_policy() -> None:
    """NaN normalization should follow the configured policy."""
    mapping = {"b": float("nan"), "a": 2.0}
    normalized = normalize_mapping(mapping, nan_policy="zero")
    expect_equal(list(normalized.keys()), ["a", "b"], label="sorted_keys")
    expect_equal(normalized["b"], 0.0, label="nan_coerced")
