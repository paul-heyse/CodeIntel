"""Tests for PR-80: batch inference produces identical schema manifests."""

from __future__ import annotations

import json

import pytest

from codeintel.build.schemas import declared_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.runtime.runtime_bundle import RuntimeBundle


def test_pr80_schema_manifest_identical_batch_vs_individual(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Ensure batch and per-table inference produce identical manifests."""
    provider = declared_schema_provider(runtime=hamilton_runtime)
    schema_index = hamilton_runtime.schema_index
    if schema_index is None:
        pytest.fail("Runtime bundle missing schema_index")

    request_batch = SchemaManifestRequest(
        targets=("risk_factors", "function_metrics"),
        infer_native=True,
        batch_infer_native=True,
        stable=True,
    )
    request_individual = SchemaManifestRequest(
        targets=("risk_factors", "function_metrics"),
        infer_native=True,
        batch_infer_native=False,
        stable=True,
    )

    context = SchemaManifestContext(
        catalog=hamilton_runtime.catalog,
        schema_index=schema_index,
        tag_query=hamilton_runtime.tag_query,
    )
    manifest_batch = compile_schema_manifest(
        provider=provider,
        context=context,
        request=request_batch,
    )
    manifest_individual = compile_schema_manifest(
        provider=provider,
        context=context,
        request=request_individual,
    )

    payload_batch = json.dumps(manifest_batch.to_json_obj(), indent=2, sort_keys=True)
    payload_individual = json.dumps(manifest_individual.to_json_obj(), indent=2, sort_keys=True)
    if payload_batch != payload_individual:
        pytest.fail("Expected batch and per-table schema manifests to be identical")
