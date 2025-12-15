"""PR-63: Schema manifest output stability."""

from __future__ import annotations

import json

import pytest

from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.schemas.provider_declared import declared_schema_provider


def test_pr63_schema_manifest_is_stable() -> None:
    """Compiling the same manifest twice should yield identical JSON."""
    provider = declared_schema_provider()
    request = SchemaManifestRequest(
        targets=("risk_factors",),
        only_native=True,
        infer_native=True,
        stable=True,
    )

    manifest_1 = compile_schema_manifest(provider=provider, request=request)
    manifest_2 = compile_schema_manifest(provider=provider, request=request)

    payload_1 = json.dumps(manifest_1.to_json_obj(), indent=2, sort_keys=True)
    payload_2 = json.dumps(manifest_2.to_json_obj(), indent=2, sort_keys=True)
    if payload_1 != payload_2:
        pytest.fail("Expected schema manifest compilation to be deterministic")
