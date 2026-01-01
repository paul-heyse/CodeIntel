"""Tests for query result coercion and streaming helpers."""

from __future__ import annotations

import math

import duckdb
import pytest

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import (
    coerce_float,
    coerce_int,
    coerce_literal,
    coerce_optional_float,
    coerce_str,
    iter_tuples_from_relation,
)
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


def test_coerce_str_accepts_bytes() -> None:
    """coerce_str converts bytes to strings."""
    expect_equal(coerce_str(b"hello", ctx="unit"), "hello", label="bytes")


def test_coerce_literal_rejects_unexpected_values() -> None:
    """coerce_literal rejects values outside the allowed set."""
    allowed = ("running", "failed")
    expect_equal(coerce_literal("running", ctx="unit", allowed=allowed), "running", label="ok")
    with pytest.raises(TypeError):
        coerce_literal("skipped", ctx="unit", allowed=allowed)


def test_iter_tuples_from_relation_streams_without_fetchall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """iter_tuples_from_relation streams batches instead of calling fetchall."""
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE tmp_rows AS SELECT range AS id FROM range(?)",
            [DEFAULT_ARROW_BATCH_SIZE + 5],
        )
        relation = con.table("tmp_rows")

        message = "fetchall should not be called"

        def _raise_fetchall() -> None:
            raise AssertionError(message)

        monkeypatch.setattr(relation, "fetchall", _raise_fetchall, raising=False)
        rows = list(iter_tuples_from_relation(relation))
        expect_equal(len(rows), DEFAULT_ARROW_BATCH_SIZE + 5, label="row_count")
    finally:
        con.close()
