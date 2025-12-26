"""Tests for build targets CLI command."""

from __future__ import annotations

import json
from typing import cast

import pytest

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import assert_success, run_cli

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


def _load_build_targets(stdout: str) -> dict[str, object]:
    payload = json.loads(stdout)
    data = payload.get("data")
    expect_is_instance(data, dict)
    return cast("dict[str, object]", data)


def test_build_targets_cli_json() -> None:
    """Build targets command returns JSON payload."""
    result = run_cli(["build", "targets", "--output-format", "json"])
    assert_success(result)

    data = _load_build_targets(result.stdout)
    outputs = data.get("outputs")
    expect_is_instance(outputs, list)
    count = data.get("count")
    expect_is_instance(count, int)
    expect_equal(len(cast("list[object]", outputs)), cast("int", count))


def test_build_targets_cli_pilot_only() -> None:
    """Pilot-only filter should return only pilot outputs."""
    result = run_cli(["build", "targets", "--pilot-only", "--output-format", "json"])
    assert_success(result)

    data = _load_build_targets(result.stdout)
    outputs = cast("list[object]", data.get("outputs"))
    expect_true(len(outputs) > 0)

    for entry in outputs:
        expect_is_instance(entry, dict)
        pilot = cast("dict[str, object]", entry).get("pilot")
        expect_true(pilot is True)


def test_build_targets_cli_table_key_filter() -> None:
    """Table key filter should constrain outputs to matching table_keys."""
    table_key = "analytics.function_metrics"
    result = run_cli(
        ["build", "targets", "--table-key", table_key, "--output-format", "json"]
    )
    assert_success(result)

    data = _load_build_targets(result.stdout)
    outputs = cast("list[object]", data.get("outputs"))
    expect_true(len(outputs) > 0)

    for entry in outputs:
        expect_is_instance(entry, dict)
        table_keys = cast("dict[str, object]", entry).get("table_keys")
        expect_is_instance(table_keys, list)
        expect_in(table_key, cast("list[object]", table_keys))
