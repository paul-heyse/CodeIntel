"""Tests for dataset specification exposure across services."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import iter_contracts
from codeintel.serving.backend import BackendLimits
from tests._helpers.datasets_assertions import (
    expect_spec_filename,
    expect_spec_has_capabilities,
    expect_spec_has_columns,
)
from tests._helpers.gateway import GatewayFactory, build_duckdb_query_service


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_dataset_specs_include_contract_fields() -> None:
    """Dataset specs should surface filenames, schema IDs, and row binding flags."""
    gateway = GatewayFactory().without_validation().open()
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
        contracts_by_key = {c.table_key: c for c in iter_contracts()}
        expected_filename = contracts_by_key.get("analytics.function_profile")
        expected_filename = expected_filename.jsonl_filename if expected_filename else None
        expect_spec_filename(profile, expected_filename)
        expect_spec_has_columns(profile)
        expect_spec_has_capabilities(profile)
    finally:
        gateway.close()
