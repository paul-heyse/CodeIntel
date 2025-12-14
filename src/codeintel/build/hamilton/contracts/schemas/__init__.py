"""Dataset schema definitions and registry.

This package is the authoritative source for dataset schemas and contract
validation in the Hamilton-first architecture.

Schemas define:
- Column names and types (via Pandera)
- Validation rules and constraints
- Row model bindings used for type checking and serialization

Examples
--------
>>> from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
>>> schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
>>> schema.column_names()  # doctest: +ELLIPSIS
(...)
"""

from __future__ import annotations

from codeintel.build.hamilton.contracts.schemas.registry import (
    SCHEMA_REGISTRY,
    DatasetSchemaRegistry,
    get_schema,
)
from codeintel.build.hamilton.contracts.schemas.schema import DatasetMetadata, DatasetSchema

__all__ = [
    "SCHEMA_REGISTRY",
    "DatasetMetadata",
    "DatasetSchema",
    "DatasetSchemaRegistry",
    "get_schema",
]
