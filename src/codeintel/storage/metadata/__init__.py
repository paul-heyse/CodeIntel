"""Metadata catalog and bootstrap utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from codeintel.storage.metadata.catalogs import (
        build_catalog_entry,
        load_canonical_catalog,
        load_latest_canonical_catalog,
        load_latest_canonical_catalog_from_connection,
        upsert_canonical_catalog,
    )
    from codeintel.storage.metadata.ddl import apply_metadata_ddl
    from codeintel.storage.metadata.sync import (
        bootstrap_metadata_datasets,
        load_derived_lineage_columns,
        sync_dataset_dataflow_graph,
        sync_derived_lineage_columns,
    )
    from codeintel.storage.metadata.validation import (
        SchemaValidationRun,
        record_schema_validation_run,
    )

_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "apply_metadata_ddl": ("codeintel.storage.metadata.ddl", "apply_metadata_ddl"),
    "bootstrap_metadata_datasets": (
        "codeintel.storage.metadata.sync",
        "bootstrap_metadata_datasets",
    ),
    "build_catalog_entry": ("codeintel.storage.metadata.catalogs", "build_catalog_entry"),
    "load_canonical_catalog": ("codeintel.storage.metadata.catalogs", "load_canonical_catalog"),
    "load_derived_lineage_columns": (
        "codeintel.storage.metadata.sync",
        "load_derived_lineage_columns",
    ),
    "load_latest_canonical_catalog": (
        "codeintel.storage.metadata.catalogs",
        "load_latest_canonical_catalog",
    ),
    "load_latest_canonical_catalog_from_connection": (
        "codeintel.storage.metadata.catalogs",
        "load_latest_canonical_catalog_from_connection",
    ),
    "sync_dataset_dataflow_graph": (
        "codeintel.storage.metadata.sync",
        "sync_dataset_dataflow_graph",
    ),
    "sync_derived_lineage_columns": (
        "codeintel.storage.metadata.sync",
        "sync_derived_lineage_columns",
    ),
    "record_schema_validation_run": (
        "codeintel.storage.metadata.validation",
        "record_schema_validation_run",
    ),
    "SchemaValidationRun": (
        "codeintel.storage.metadata.validation",
        "SchemaValidationRun",
    ),
    "upsert_canonical_catalog": ("codeintel.storage.metadata.catalogs", "upsert_canonical_catalog"),
}


def __getattr__(name: str) -> object:
    entry = _EXPORTS.get(name)
    if entry is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attr_name = entry
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__: tuple[str, ...] = (
    "SchemaValidationRun",
    "apply_metadata_ddl",
    "bootstrap_metadata_datasets",
    "build_catalog_entry",
    "load_canonical_catalog",
    "load_derived_lineage_columns",
    "load_latest_canonical_catalog",
    "load_latest_canonical_catalog_from_connection",
    "record_schema_validation_run",
    "sync_dataset_dataflow_graph",
    "sync_derived_lineage_columns",
    "upsert_canonical_catalog",
)
