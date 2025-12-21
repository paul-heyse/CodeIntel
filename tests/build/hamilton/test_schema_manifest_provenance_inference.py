"""Tests for manifest provenance and inference status."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest


def test_manifest_provenance_includes_inference_status() -> None:
    """Manifest provenance should include inference status for tables."""
    provider = get_schema_provider()
    request = SchemaManifestRequest(
        targets=("modules",),
        include_provenance=True,
    )
    manifest = compile_schema_manifest(provider=provider, request=request)
    provenance = manifest.table_provenance.get("core.modules")
    if provenance is None:
        pytest.fail("Expected table provenance for core.modules.")
    if provenance.inference_status != "override":
        pytest.fail("Expected core.modules to report override inference status.")
