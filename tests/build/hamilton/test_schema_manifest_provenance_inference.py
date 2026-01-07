"""Tests for manifest provenance and inference status."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


def test_manifest_provenance_includes_inference_status(
    hamilton_runtime: HamiltonRuntimeBundle,
) -> None:
    """Manifest provenance should include inference status for tables."""
    provider = get_schema_provider()
    schema_index = hamilton_runtime.schema_index
    if schema_index is None:
        pytest.fail("Runtime bundle missing schema_index")
    request = SchemaManifestRequest(
        targets=("modules",),
        include_provenance=True,
    )
    manifest = compile_schema_manifest(
        provider=provider,
        context=SchemaManifestContext(
            catalog=hamilton_runtime.catalog,
            schema_index=schema_index,
            tag_query=hamilton_runtime.tag_query,
        ),
        request=request,
    )
    provenance = manifest.table_provenance.get("core.modules")
    if provenance is None:
        pytest.fail("Expected table provenance for core.modules.")
    if provenance.inference_status != "override":
        pytest.fail("Expected core.modules to report override inference status.")
