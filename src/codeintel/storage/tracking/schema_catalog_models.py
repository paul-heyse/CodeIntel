"""Shared models for schema catalog persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

DEFAULT_SCHEMA_MANIFEST_KIND = "schema_manifest_v2"


@dataclass(frozen=True)
class SchemaVersionRecord:
    """Record of a content-addressed schema version."""

    schema_digest: str
    schema_hash: str
    schema_json: dict[str, object]
    renderer_cache: dict[str, object] | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaRegistryRecord:
    """Current schema pointer for a table key."""

    table_key: str
    schema_digest: str
    schema_hash: str
    derivation_kind: str
    derivation_source: str
    inference_status: str | None = None
    inference_error: str | None = None
    catalog_hash: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class SchemaManifestRunRecord:
    """Schema manifest catalog linkage for a build run."""

    run_id: str
    repo: str
    commit: str
    manifest_kind: str
    catalog_hash: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaOverrideVersionRecord:
    """Record of a schema override version for an inferable table."""

    version_id: str
    table_key: str
    schema_digest: str
    schema_hash: str
    catalog_hash: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaOverrideRegistryRecord:
    """Current override pointer for an inferable table."""

    table_key: str
    schema_digest: str
    schema_hash: str
    version_id: str
    updated_at: datetime | None = None


@dataclass(frozen=True)
class OverrideRegistryRefreshResult:
    """Summary of an override registry refresh attempt."""

    status: str
    reason: str | None
    version_id: str | None
    tables: int
    schema_versions_rows: int
    override_versions_rows: int
    override_registry_rows: int


@dataclass(frozen=True)
class SchemaCatalogRequest:
    """Inputs for compiling or persisting a schema catalog."""

    run_id: str
    repo: str
    commit: str
    catalog_inputs: Mapping[str, object] | None = None
    include_views: bool = True
    strict_provenance: bool = True
    strict_hash_match: bool = True
    now: datetime | None = None
    catalog_kind: str = DEFAULT_SCHEMA_MANIFEST_KIND
    manifest_kind: str = DEFAULT_SCHEMA_MANIFEST_KIND


__all__ = [
    "DEFAULT_SCHEMA_MANIFEST_KIND",
    "OverrideRegistryRefreshResult",
    "SchemaCatalogRequest",
    "SchemaManifestRunRecord",
    "SchemaVersionRecord",
    "TableSchemaOverrideRegistryRecord",
    "TableSchemaOverrideVersionRecord",
    "TableSchemaRegistryRecord",
]
