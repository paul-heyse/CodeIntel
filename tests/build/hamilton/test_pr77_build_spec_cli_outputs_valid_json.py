"""Tests for PR-77: BuildSpec CLI surfaces."""

from __future__ import annotations

import json
import re

import pytest

from tests._helpers.cli import assert_success, run_cli


def _require(*, condition: bool, message: str) -> None:
    """Assert condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def test_build_spec_cli_compile_outputs_valid_json() -> None:
    """Verify `codeintel build spec compile` prints valid BuildSpec JSON."""
    result = run_cli(
        ["build", "spec", "compile", "--format", "json"],
        env={"CODEINTEL_LOG_LEVEL": "WARNING"},
    )
    assert_success(result)

    payload = json.loads(result.stdout)
    _require(condition=isinstance(payload, dict), message="Expected top-level JSON object")
    if not isinstance(payload, dict):
        return

    spec_version = payload.get("spec_version")
    _require(condition=isinstance(spec_version, int), message="spec_version must be an int")
    _require(
        condition=isinstance(spec_version, int) and spec_version >= 1,
        message=f"spec_version must be >= 1, got {spec_version!r}",
    )

    targets = payload.get("targets")
    _require(condition=isinstance(targets, list), message="targets must be a list")
    _require(
        condition=isinstance(targets, list) and bool(targets), message="targets must be non-empty"
    )

    datasets = payload.get("datasets")
    _require(condition=isinstance(datasets, list), message="datasets must be a list")

    buildspec_hash = payload.get("buildspec_hash")
    _require(condition=isinstance(buildspec_hash, str), message="buildspec_hash must be a string")
    _require(
        condition=isinstance(buildspec_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", buildspec_hash) is not None,
        message=f"buildspec_hash must be 64 lowercase hex chars, got {buildspec_hash!r}",
    )
