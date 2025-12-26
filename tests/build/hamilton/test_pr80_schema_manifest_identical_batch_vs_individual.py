"""Tests for PR-80: batch inference produces identical schema manifests."""

from __future__ import annotations

import json

import pytest

from codeintel.build.schemas import declared_schema_provider
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest


def test_pr80_schema_manifest_identical_batch_vs_individual() -> None:
    """Ensure batch and per-table inference produce identical manifests."""
    provider = declared_schema_provider()

    request_batch = SchemaManifestRequest(
        targets=("risk_factors", "hotspots"),
        infer_native=True,
        batch_infer_native=True,
        stable=True,
    )
    request_individual = SchemaManifestRequest(
        targets=("risk_factors", "hotspots"),
        infer_native=True,
        batch_infer_native=False,
        stable=True,
    )

    manifest_batch = compile_schema_manifest(provider=provider, request=request_batch)
    manifest_individual = compile_schema_manifest(provider=provider, request=request_individual)

    payload_batch = json.dumps(manifest_batch.to_json_obj(), indent=2, sort_keys=True)
    payload_individual = json.dumps(manifest_individual.to_json_obj(), indent=2, sort_keys=True)
    if payload_batch != payload_individual:
        pytest.fail("Expected batch and per-table schema manifests to be identical")
