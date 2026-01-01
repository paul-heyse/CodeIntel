"""Backward-compatible re-exports for schema catalog models."""

from __future__ import annotations

from codeintel.core.schemas.schema_catalog_models import (
    DEFAULT_SCHEMA_MANIFEST_KIND,
    ColumnStatsEntry,
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    OverrideRegistryRefreshResult,
    ParquetStatsPayload,
    SchemaCatalogRequest,
    SchemaManifestRunRecord,
    SchemaObservationRecord,
    SchemaVersionRecord,
    TableSchemaOverrideRegistryRecord,
    TableSchemaOverrideVersionRecord,
    TableSchemaRegistryRecord,
)

__all__ = [
    "DEFAULT_SCHEMA_MANIFEST_KIND",
    "ColumnStatsEntry",
    "ColumnStatsPayload",
    "DatasetStatsPayload",
    "DerivedSettingsPayload",
    "OverrideRegistryRefreshResult",
    "ParquetStatsPayload",
    "SchemaCatalogRequest",
    "SchemaManifestRunRecord",
    "SchemaObservationRecord",
    "SchemaVersionRecord",
    "TableSchemaOverrideRegistryRecord",
    "TableSchemaOverrideVersionRecord",
    "TableSchemaRegistryRecord",
]
