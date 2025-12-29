"""Tests for observation payload encoding and decoding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.tracking.observation_codec import (
    decode_column_stats,
    decode_dataset_stats,
    decode_derived_settings,
    encode_column_stats,
    encode_dataset_stats,
    encode_derived_settings,
)

pytestmark = pytest.mark.no_runtime_env


@dataclass(frozen=True)
class _Stats:
    null_count: int
    non_null_count: int
    distinct_max: int | None
    min_value: object | None
    max_value: object | None
    length_sum: int
    length_count: int


def test_encode_column_stats_payload() -> None:
    stats = {
        "name": _Stats(
            null_count=0,
            non_null_count=5,
            distinct_max=3,
            min_value=1,
            max_value=9,
            length_sum=20,
            length_count=5,
        )
    }
    payload = encode_column_stats(stats)
    assert payload is not None
    entry = payload["name"]
    assert entry["null_count"] == 0
    assert entry["non_null_count"] == 5
    assert entry["distinct_count_max"] == 3
    assert entry["min"] == 1
    assert entry["max"] == 9
    assert entry["avg_length"] == 4


def test_encode_dataset_stats_payload() -> None:
    payload = encode_dataset_stats(
        row_count=5,
        batch_count=1,
        total_bytes=10,
        manifest_stats={"row_groups": 2},
        manifest_row_count=5,
    )
    assert payload["row_count"] == 5
    assert payload["batch_count"] == 1
    assert payload["total_bytes"] == 10
    assert payload["manifest_row_count"] == 5
    assert payload["parquet_stats"] == {"row_groups": 2}


def test_encode_derived_settings_payload() -> None:
    schema = TableSchema(
        schema="analytics",
        name="demo",
        columns=[Column("name", "VARCHAR", nullable=True)],
    )
    stats = {
        "name": _Stats(
            null_count=0,
            non_null_count=100,
            distinct_max=2,
            min_value=None,
            max_value=None,
            length_sum=0,
            length_count=0,
        )
    }
    payload = encode_derived_settings(
        table_schema=schema,
        column_stats=stats,
        row_count=100,
        total_bytes=10_000,
        extras_policy="retain",
    )
    assert payload is not None
    assert payload["extras_policy"] == "retain"
    assert payload["dictionary_encode_columns"] == ["name"]
    assert payload["dictionary_max_cardinality"] == 2
    assert payload["unify_dictionaries"] is True


def test_decode_column_stats_valid_payload() -> None:
    payload = {"name": {"null_count": 0, "non_null_count": 3, "distinct_count_max": 2}}
    decoded = decode_column_stats(payload)
    assert decoded == payload


def test_decode_column_stats_invalid_payload() -> None:
    payload = {"name": {"null_count": "nope"}}
    assert decode_column_stats(payload) is None


def test_decode_dataset_stats_invalid_payload() -> None:
    payload = {"row_count": "bad"}
    assert decode_dataset_stats(payload) is None


def test_decode_derived_settings_valid_payload() -> None:
    payload = {
        "extras_policy": "retain",
        "dictionary_encode_columns": ["a"],
        "dictionary_max_cardinality": 10,
        "unify_dictionaries": True,
        "avg_row_bytes": 12.5,
    }
    assert decode_derived_settings(payload) == payload


def test_decode_derived_settings_invalid_payload() -> None:
    payload = {"extras_policy": 12}
    assert decode_derived_settings(payload) is None
