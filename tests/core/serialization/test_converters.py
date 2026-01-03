"""Tests for core serialization converters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from codeintel.core.serialization.converters import deserialize_value


def test_deserialize_value_optional_datetime() -> None:
    """Optional datetime should deserialize when value is present."""
    raw = "2024-01-01T12:00:00"
    result = deserialize_value(raw, datetime | None)
    if not isinstance(result, datetime):
        pytest.fail(f"Expected datetime, got {type(result)}")
    if result != datetime.fromisoformat(raw):
        pytest.fail(f"Unexpected datetime value: {result}")


def test_deserialize_value_optional_datetime_none() -> None:
    """Optional datetime should remain None when value is None."""
    result = deserialize_value(None, datetime | None)
    if result is not None:
        pytest.fail("Expected None for optional datetime")


def test_deserialize_value_optional_path() -> None:
    """Optional Path should deserialize into a Path instance."""
    raw = "/tmp/demo"
    result = deserialize_value(raw, Path | None)
    if not isinstance(result, Path):
        pytest.fail(f"Expected Path, got {type(result)}")
    if str(result) != raw:
        pytest.fail(f"Unexpected path value: {result}")


def test_deserialize_value_union_multiple_types_returns_raw() -> None:
    """Union with multiple concrete types should return raw value."""
    raw = "2024-01-01T12:00:00"
    result = deserialize_value(raw, datetime | Path)
    if result != raw:
        pytest.fail(f"Expected raw value, got {result}")
