"""Phase 4 asset catalog helpers (versions, lineage, diffs, aliases)."""

from __future__ import annotations

from codeintel.build.assets.emitter import persist_asset_catalog_for_run
from codeintel.build.assets.fingerprinting import (
    DEFAULT_FINGERPRINT_POLICY,
    ArtifactVersionInput,
    FingerprintMode,
    FingerprintPolicy,
    TableVersionInput,
    compute_fast_version_hash,
    compute_table_schema_hash,
)
from codeintel.build.assets.impact import (
    ImpactedAsset,
    ImpactResult,
    compute_impact,
)

__all__ = [
    "DEFAULT_FINGERPRINT_POLICY",
    "ArtifactVersionInput",
    "FingerprintMode",
    "FingerprintPolicy",
    "ImpactResult",
    "ImpactedAsset",
    "TableVersionInput",
    "compute_fast_version_hash",
    "compute_impact",
    "compute_table_schema_hash",
    "persist_asset_catalog_for_run",
]
