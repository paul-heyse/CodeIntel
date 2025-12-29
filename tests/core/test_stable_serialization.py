"""Tests for stable serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from codeintel.core.serialization.stable import stable_json_value, stable_stringify
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance


def test_stable_stringify_sorts_dict_keys() -> None:
    """Ensure dict keys are deterministic in the serialized output."""
    payload = {"b": 1, "a": 2}
    serialized = stable_stringify(payload)
    expect_equal(serialized, '{"a":2,"b":1}')


def test_stable_stringify_sorts_set_values() -> None:
    """Ensure sets are serialized deterministically."""
    serialized = stable_stringify({"b", "a"})
    expect_equal(serialized, '["a","b"]')


def test_stable_json_value_omits_private_and_none_fields() -> None:
    """Ensure omit flags drop private fields and None values."""

    @dataclass(frozen=True)
    class Sample:
        name: str
        _secret: str
        optional: str | None = None

    value = Sample(name="alpha", _secret="hidden", optional=None)
    serialized = stable_json_value(
        value,
        omit_none=True,
        omit_private_fields=True,
    )
    expect_equal(serialized, {"name": "alpha"})


def test_stable_json_value_handles_path_and_datetime() -> None:
    """Ensure Path and datetime values serialize as strings."""
    payload = {"path": Path("src/file.py"), "ts": datetime(2024, 1, 1, 0, 0, 0)}
    serialized = stable_json_value(payload)
    expect_is_instance(serialized, dict)
    expect_in("src/file.py", str(serialized))
