"""Tests for dataset specification exposure across services."""

from __future__ import annotations

import pytest

from codeintel.config.datasets import DEFAULT_JSONL_FILENAMES
from codeintel.serving.backend import BackendLimits
from tests._helpers.gateway import build_duckdb_query_service, gateway_with_macros


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_dataset_specs_include_contract_fields() -> None:
    """Dataset specs should surface filenames, schema IDs, and row binding flags."""
    gateway = gateway_with_macros(validate_schema=False)
    try:
        query = build_duckdb_query_service(
            gateway=gateway, repo="repo", commit="commit", limits=BackendLimits()
        )
        specs = query.datasets.dataset_specs()
        spec_map = {spec.name: spec for spec in specs}
        _require(condition="function_profile" in spec_map, message="function_profile spec missing")
        profile = spec_map["function_profile"]
        _require(
            condition=profile.json_schema_id == "function_profile",
            message="json_schema_id missing for function_profile",
        )
        _require(
            condition=profile.has_row_binding is True,
            message="Row binding flag missing for function_profile",
        )
        expected_filename = DEFAULT_JSONL_FILENAMES.get("analytics.function_profile")
        _require(
            condition=profile.jsonl_filename == expected_filename,
            message="jsonl_filename mismatch for function_profile",
        )
        _require(
            condition=bool(profile.schema_columns) is True,
            message="Schema columns missing for function_profile",
        )
    finally:
        gateway.close()
