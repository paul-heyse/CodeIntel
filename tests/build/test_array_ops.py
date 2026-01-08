"""Tests for array operation helpers."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.tabular.array_ops import take_by_key
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_take_by_key_missing_policy_error() -> None:
    """take_by_key should raise when keys are missing and missing_policy is error."""
    keys = pa.array(["a", "c"])
    key_set = pa.array(["a", "b"])
    values = pa.array([1, 2])

    with pytest.raises(ValueError, match="missing keys"):
        _ = take_by_key(keys, key_set, values, missing_policy="error")


def test_take_by_key_missing_policy_null() -> None:
    """take_by_key should return nulls when keys are missing and missing_policy is null."""
    keys = pa.array(["a", "c"])
    key_set = pa.array(["a", "b"])
    values = pa.array([1, 2])

    result = take_by_key(keys, key_set, values, missing_policy="null")

    expect_equal(result.to_pylist(), [1, None])
