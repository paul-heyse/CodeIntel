"""Contract primitives for dataset metadata.

This module defines the core dataclasses used for dataset contracts.
These are intentionally lightweight and have no dependencies on other
parts of the config/datasets infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema, TableSchema
    from codeintel.core.schemas.row_models import GeneratedRowBinding


@dataclass(frozen=True)
class DatasetContract:
    """Metadata describing a logical dataset backed by a DuckDB table or view.

    Parameters
    ----------
    table_key
        Fully qualified DuckDB identifier, e.g. "analytics.function_profile".
    name
        Logical dataset name, e.g. "function_profile".
    schema
        Statically defined TableSchema when the dataset is backed by a table;
        None when the dataset is a view.
    row_binding
        Optional binding to a schema-generated row model and serializer.
    json_schema_id
        Optional JSON Schema identifier (without .json) used for export validation.
    jsonl_filename
        Default filename for JSONL exports (may be None when not exported).
    parquet_filename
        Default filename for Parquet exports (may be None when not exported).
    is_view
        True when this dataset is a docs.* view instead of a base table.
    owner_package
        Optional package ownership derived from schema prefix.
    tags
        Classification tags applied to the dataset.
    description
        Optional human-readable description.
    family
        Optional dataset family inferred from schema prefix.
    owner
        Optional team or individual owner.
    freshness_sla
        Optional freshness expectation (e.g., "daily", "hourly").
    retention_policy
        Optional retention policy descriptor (e.g., "90d").
    stable_id
        Optional stable identifier for comparing contracts across versions.
    schema_version
        Optional schema version string for change tracking.
    upstream_dependencies
        Optional tuple of other dataset names this dataset depends on.
    validation_profile
        Validation strictness level ("strict" or "lenient").
    composition
        Optional CompositeSchema for profile datasets.
    deprecated
        Whether this dataset is deprecated.
    deprecation_message
        Message explaining deprecation and migration path.
    """

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: GeneratedRowBinding | None = None
    json_schema_id: str | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False
    owner_package: Literal["core", "analytics", "graphs", "qa", "docs"] | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str | None = None
    family: str | None = None
    owner: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    validation_profile: Literal["strict", "lenient"] = "strict"
    composition: CompositeSchema | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    def has_row_binding(self) -> bool:
        """Return True when this dataset has a TypedDict row binding.

        Returns
        -------
        bool
            True if row_binding is not None.
        """
        return self.row_binding is not None

    def require_row_binding(self) -> GeneratedRowBinding:
        """Return the row binding or raise a clear error if missing.

        Returns
        -------
        GeneratedRowBinding
            Configured row binding for this dataset.

        Raises
        ------
        KeyError
            If no row binding is configured for this dataset.
        """
        if self.row_binding is None:
            message = f"Dataset {self.name} ({self.table_key}) has no row binding"
            raise KeyError(message)
        return self.row_binding

    def capabilities(self) -> dict[str, bool]:
        """Return capability flags derived from attached metadata.

        Returns
        -------
        dict[str, bool]
            Flags for validation and export support.
        """
        docs_view = self.table_key.startswith("docs.")
        read_only = self.is_view or docs_view or "read_only" in self.tags
        return {
            "can_validate": self.json_schema_id is not None,
            "can_export_jsonl": self.jsonl_filename is not None,
            "can_export_parquet": self.parquet_filename is not None,
            "has_row_binding": self.row_binding is not None,
            "is_view": self.is_view,
            "docs_view": docs_view,
            "read_only": read_only,
            "dataset_rows_only": "dataset_rows_only" in self.tags,
        }

    def column_names(self) -> tuple[str, ...]:
        """Return column names in schema definition order.

        Returns
        -------
        tuple[str, ...]
            Ordered column names, or empty tuple for views without schema.
        """
        if self.schema is None:
            return ()
        return tuple(self.schema.column_names())


__all__ = [
    "DatasetContract",
]
