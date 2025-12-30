"""Iceberg catalog, schema, and scan planning helpers."""

from __future__ import annotations

from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.guardrails import (
    IcebergGuardrailError,
    iceberg_enforced_table,
    require_iceberg_read,
    require_iceberg_write,
)
from codeintel.core.iceberg.scan_plan import IcebergScanPlan
from codeintel.core.iceberg.schema import (
    arrow_schema_with_iceberg_ids,
    iceberg_schema_to_arrow_schema,
    name_mapping_from_arrow_schema,
    table_schema_to_iceberg_schema,
)
from codeintel.core.iceberg.stream import IcebergColumnarStream

__all__ = [
    "IcebergCatalogProvider",
    "IcebergColumnarStream",
    "IcebergGuardrailError",
    "IcebergScanPlan",
    "arrow_schema_with_iceberg_ids",
    "iceberg_enforced_table",
    "iceberg_schema_to_arrow_schema",
    "name_mapping_from_arrow_schema",
    "require_iceberg_read",
    "require_iceberg_write",
    "table_schema_to_iceberg_schema",
]
