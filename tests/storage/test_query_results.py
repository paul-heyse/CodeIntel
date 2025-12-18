"""Tests for scalar result coercion helpers."""

from __future__ import annotations

import math

import pytest

from codeintel.storage.query_results import coerce_float, coerce_int, coerce_optional_float
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_is_none


def test_coerce_int_accepts_int_like_values() -> None:
    """coerce_int converts supported scalar types to int."""
    expect_equal(coerce_int(5, ctx="unit"), 5, label="int")
    expect_equal(coerce_int(5.0, ctx="unit"), 5, label="float integral")
    expect_equal(coerce_int("6", ctx="unit"), 6, label="string digits")
    expect_equal(coerce_int("-7", ctx="unit"), -7, label="string negative")


def test_coerce_int_rejects_non_integral_float() -> None:
    """coerce_int rejects non-integral floats."""
    with pytest.raises(TypeError):
        coerce_int(1.25, ctx="unit")


def test_coerce_float_accepts_numeric_values() -> None:
    """coerce_float converts supported scalar types to float."""
    expect_equal(coerce_float(5, ctx="unit"), 5.0, label="int")
    expect_equal(coerce_float(5.25, ctx="unit"), 5.25, label="float")
    expect_equal(coerce_float("6.5", ctx="unit"), 6.5, label="string")


def test_coerce_optional_float_treats_nan_as_missing() -> None:
    """coerce_optional_float returns None for NaN values."""
    expect_is_none(coerce_optional_float(None, ctx="unit"), label="none")
    expect_is_none(coerce_optional_float(float("nan"), ctx="unit"), label="nan")
    expect_equal(coerce_optional_float(math.pi, ctx="unit"), math.pi, label="pi")
