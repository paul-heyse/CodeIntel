"""Tests for registry CLI commands."""

from __future__ import annotations

import json
from typing import cast

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import assert_success, run_cli


def _load_registry_outputs(stdout: str) -> dict[str, object]:
    payload = json.loads(stdout)
    data = payload.get("data")
    expect_is_instance(data, dict)
    return cast("dict[str, object]", data)


def _load_registry_tools(stdout: str) -> dict[str, object]:
    payload = json.loads(stdout)
    data = payload.get("data")
    expect_is_instance(data, dict)
    return cast("dict[str, object]", data)


def test_registry_outputs_cli_json() -> None:
    """Registry outputs command returns JSON payload."""
    result = run_cli(["registry", "outputs", "--output-format", "json"])
    assert_success(result)

    data = _load_registry_outputs(result.stdout)
    outputs = data.get("outputs")
    expect_is_instance(outputs, list)
    count = data.get("count")
    expect_is_instance(count, int)
    expect_equal(len(cast("list[object]", outputs)), cast("int", count))


def test_registry_outputs_cli_pilot_only() -> None:
    """Pilot-only filter should return only pilot outputs."""
    result = run_cli(["registry", "outputs", "--pilot-only", "--output-format", "json"])
    assert_success(result)

    data = _load_registry_outputs(result.stdout)
    outputs = cast("list[object]", data.get("outputs"))
    expect_true(len(outputs) > 0)

    for entry in outputs:
        expect_is_instance(entry, dict)
        pilot = cast("dict[str, object]", entry).get("pilot")
        expect_true(pilot is True)


def test_registry_validate_cli_json() -> None:
    """Registry validate command returns JSON payload."""
    result = run_cli(["registry", "validate", "--output-format", "json"])
    assert_success(result)

    payload = json.loads(result.stdout)
    data = payload.get("data")
    expect_is_instance(data, dict)
    output_count = cast("dict[str, object]", data).get("output_count")
    expect_is_instance(output_count, int)


def test_registry_tools_cli_json() -> None:
    """Registry tools command returns JSON payload."""
    result = run_cli(["registry", "tools", "--output-format", "json"])
    assert_success(result)

    data = _load_registry_tools(result.stdout)
    tools = data.get("tools")
    expect_is_instance(tools, list)
    count = data.get("count")
    expect_is_instance(count, int)
    missing_count = data.get("missing_count")
    expect_is_instance(missing_count, int)
    expect_equal(len(cast("list[object]", tools)), cast("int", count))
