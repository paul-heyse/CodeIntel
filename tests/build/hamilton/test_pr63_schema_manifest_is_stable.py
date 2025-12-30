"""PR-63: Schema manifest output stability."""

from __future__ import annotations

import json

import pytest

from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.schemas.provider_unified import unified_schema_provider
from codeintel.runtime.runtime_bundle import RuntimeBundle
from codeintel.storage.views.inventory import discover_derived_docs_views


def test_pr63_schema_manifest_is_stable(hamilton_runtime: RuntimeBundle) -> None:
    """Compiling the same manifest twice should yield identical JSON."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    request = SchemaManifestRequest(
        targets=("risk_factors",),
        infer_native=True,
        stable=True,
    )

    schema_index = hamilton_runtime.schema_index
    if schema_index is None:
        pytest.fail("Runtime bundle missing schema_index")
    context = SchemaManifestContext(
        catalog=hamilton_runtime.catalog,
        schema_index=schema_index,
        tag_query=hamilton_runtime.tag_query,
    )
    manifest_1 = compile_schema_manifest(
        provider=provider,
        context=context,
        request=request,
    )
    manifest_2 = compile_schema_manifest(
        provider=provider,
        context=context,
        request=request,
    )

    payload_1 = json.dumps(manifest_1.to_json_obj(), indent=2, sort_keys=True)
    payload_2 = json.dumps(manifest_2.to_json_obj(), indent=2, sort_keys=True)
    if payload_1 != payload_2:
        pytest.fail("Expected schema manifest compilation to be deterministic")


def test_schema_manifest_includes_derived_views(hamilton_runtime: RuntimeBundle) -> None:
    """Derived docs views should appear when include_views=True."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    schema_index = hamilton_runtime.schema_index
    if schema_index is None:
        pytest.fail("Runtime bundle missing schema_index")
    context = SchemaManifestContext(
        catalog=hamilton_runtime.catalog,
        schema_index=schema_index,
        tag_query=hamilton_runtime.tag_query,
    )
    request = SchemaManifestRequest(all_targets=True, include_views=True, stable=True)
    manifest = compile_schema_manifest(
        provider=provider,
        context=context,
        request=request,
    )
    manifest_view_keys = {view.table_key for view in manifest.views}
    derived_view_keys = set(discover_derived_docs_views())
    if not manifest_view_keys.intersection(derived_view_keys):
        pytest.fail("Expected schema manifest to include derived docs views")
