"""Compatibility wrapper for Arrow schema rendering helpers."""

from __future__ import annotations

from codeintel.core.schemas.contracts import (
    ARROW_FIELD_METADATA_KEYS,
    ARROW_SCHEMA_CONTRACT_VERSION,
    ARROW_SCHEMA_METADATA_KEYS,
    DEFAULT_EXTRAS_COLUMN,
    DEFAULT_EXTRAS_POLICY,
    EXTRAS_POLICIES,
    ArrowSchemaMetadata,
    ArrowSchemaProvenance,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
    arrow_schema_from_table_schema,
)

__all__ = [
    "ARROW_FIELD_METADATA_KEYS",
    "ARROW_SCHEMA_CONTRACT_VERSION",
    "ARROW_SCHEMA_METADATA_KEYS",
    "DEFAULT_EXTRAS_COLUMN",
    "DEFAULT_EXTRAS_POLICY",
    "EXTRAS_POLICIES",
    "ArrowSchemaMetadata",
    "ArrowSchemaProvenance",
    "ExtrasPolicy",
    "arrow_contract_for_table_schema",
    "arrow_schema_from_table_schema",
]
