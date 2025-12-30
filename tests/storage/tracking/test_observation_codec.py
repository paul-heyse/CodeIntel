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

EXPECTED_NON_NULL_COUNT = 5
EXPECTED_DISTINCT_COUNT_MAX = 3
EXPECTED_NULL_COUNT = 0
EXPECTED_MIN_VALUE = 1
EXPECTED_BATCH_COUNT = 1
EXPECTED_ROW_GROUPS = 2
EXPECTED_MAX_VALUE = 9
EXPECTED_AVG_LENGTH = 4
EXPECTED_ROW_COUNT = 5
EXPECTED_TOTAL_BYTES = 10
EXPECTED_MANIFEST_ROW_COUNT = 5
EXPECTED_DICTIONARY_MAX_CARDINALITY = 2


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
    """Encode column stats payload with numeric aggregates."""
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
    assert entry.get("null_count") == EXPECTED_NULL_COUNT
    assert entry.get("non_null_count") == EXPECTED_NON_NULL_COUNT
    assert entry.get("distinct_count_max") == EXPECTED_DISTINCT_COUNT_MAX
    assert entry.get("min") == EXPECTED_MIN_VALUE
    assert entry.get("max") == EXPECTED_MAX_VALUE
    assert entry.get("avg_length") == EXPECTED_AVG_LENGTH


def test_encode_dataset_stats_payload() -> None:
    """Encode dataset stats payload with manifest metadata."""
    payload = encode_dataset_stats(
        row_count=5,
        batch_count=EXPECTED_BATCH_COUNT,
        total_bytes=EXPECTED_TOTAL_BYTES,
        manifest_stats={"row_groups": EXPECTED_ROW_GROUPS},
        manifest_row_count=5,
    )
    assert payload.get("row_count") == EXPECTED_ROW_COUNT
    assert payload.get("batch_count") == EXPECTED_BATCH_COUNT
    assert payload.get("total_bytes") == EXPECTED_TOTAL_BYTES
    assert payload.get("manifest_row_count") == EXPECTED_MANIFEST_ROW_COUNT
    assert payload.get("parquet_stats") == {"row_groups": EXPECTED_ROW_GROUPS}


def test_encode_derived_settings_payload() -> None:
    """Encode derived settings payload from observed stats."""
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
    assert payload.get("extras_policy") == "retain"
    assert payload.get("dictionary_encode_columns") == ["name"]
    assert payload.get("dictionary_max_cardinality") == EXPECTED_DICTIONARY_MAX_CARDINALITY
    assert payload.get("unify_dictionaries") is True


def test_decode_column_stats_valid_payload() -> None:
    """Decode valid column stats payload."""
    payload = {"name": {"null_count": 0, "non_null_count": 3, "distinct_count_max": 2}}
    decoded = decode_column_stats(payload)
    assert decoded == payload


def test_decode_column_stats_invalid_payload() -> None:
    """Reject invalid column stats payloads."""
    payload = {"name": {"null_count": "nope"}}
    assert decode_column_stats(payload) is None


def test_decode_dataset_stats_invalid_payload() -> None:
    """Reject invalid dataset stats payloads."""
    payload = {"row_count": "bad"}
    assert decode_dataset_stats(payload) is None


def test_decode_derived_settings_valid_payload() -> None:
    """Decode valid derived settings payload."""
    payload = {
        "extras_policy": "retain",
        "dictionary_encode_columns": ["a"],
        "dictionary_max_cardinality": 10,
        "unify_dictionaries": True,
        "avg_row_bytes": 12.5,
    }
    assert decode_derived_settings(payload) == payload


def test_decode_derived_settings_invalid_payload() -> None:
    """Reject invalid derived settings payloads."""
    payload = {"extras_policy": 12}
    assert decode_derived_settings(payload) is None
