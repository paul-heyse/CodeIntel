"""Compatibility wrappers for Hamilton-backed schema inference.

This module retains the legacy provider_hamilton API but delegates to the
canonical SchemaInferenceService in ``codeintel.build.schemas.inference_service``.
"""

from __future__ import annotations

from codeintel.build.schemas.inference_service import (
    HamiltonSchemaProvider,
    SchemaInferenceService,
    get_schema_inference_service,
    infer_schema_for_table_key,
    infer_table_schemas,
    inferable_native_table_keys,
)

__all__ = [
    "HamiltonSchemaProvider",
    "SchemaInferenceService",
    "get_schema_inference_service",
    "infer_schema_for_table_key",
    "infer_table_schemas",
    "inferable_native_table_keys",
]
