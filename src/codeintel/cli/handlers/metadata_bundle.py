"""Helpers for ingesting build metadata bundles into storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from codeintel.core.execution.ids import new_run_id
from codeintel.core.manifests import SchemaManifest, read_manifest_json
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.metadata.ingest import (
    BundleIngestReport,
    BundleManifest,
    bundle_manifest_from_path,
    load_build_metadata_bundle,
)
from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking
from codeintel.storage.tracking.schema_catalog_models import (
    OverrideRegistryRefreshResult,
    SchemaCatalogRequest,
)


@dataclass(frozen=True, slots=True)
class BundleIngestRequest:
    """Inputs required to ingest a build metadata bundle."""

    bundle_root: Path
    db_path: Path
    repo: str
    commit: str
    run_id: str | None = None
    catalog_inputs: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BundleIngestOutcome:
    """Result payload for build metadata bundle ingestion."""

    bundle_manifest: BundleManifest
    report: BundleIngestReport
    run_id: str
    override_refresh: OverrideRegistryRefreshResult | None
    renderer_cache_rows: int


def ingest_metadata_bundle(request: BundleIngestRequest) -> BundleIngestOutcome:
    """Ingest a build metadata bundle and refresh schema override state.

    Parameters
    ----------
    request
        Bundle ingest request parameters.

    Returns
    -------
    BundleIngestOutcome
        Bundle ingest results and override refresh metadata.
    """
    bundle_manifest = bundle_manifest_from_path(request.bundle_root)
    resolved_run_id = request.run_id or bundle_manifest.run_id or new_run_id("meta")
    config = StorageConfig(
        db_path=request.db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        repo=request.repo,
        commit=request.commit,
    )
    session = DuckDBSession(config)
    override_refresh: OverrideRegistryRefreshResult | None = None
    renderer_cache_rows = 0
    with session.connect() as con:
        report = load_build_metadata_bundle(request.bundle_root, con)
        schema_manifest_path = request.bundle_root / "schema" / "schema_manifest.json"
        if schema_manifest_path.is_file():
            schema_manifest = read_manifest_json(schema_manifest_path, payload_type=SchemaManifest)
            tracker = SchemaCatalogTracking(MinimalStorageGateway(con, config=config))
            schema_request = SchemaCatalogRequest(
                run_id=resolved_run_id,
                repo=request.repo,
                commit=request.commit,
                catalog_inputs=request.catalog_inputs,
                include_views=True,
                strict_provenance=True,
                strict_hash_match=True,
            )
            override_refresh = tracker.refresh_override_registry_from_manifest(
                schema_manifest,
                request=schema_request,
                catalog_hash=report.schema_manifest_hash,
            )
            renderer_cache_rows = tracker.backfill_renderer_cache(
                schema_manifest,
                include_views=schema_request.include_views,
            )

    return BundleIngestOutcome(
        bundle_manifest=bundle_manifest,
        report=report,
        run_id=resolved_run_id,
        override_refresh=override_refresh,
        renderer_cache_rows=renderer_cache_rows,
    )
