"""Canonical catalog utilities for build-time registries."""

from __future__ import annotations

from codeintel.build.catalogs.canonical import (
    CONTRACT_CATALOG_KIND,
    TARGET_CATALOG_KIND,
    load_contract_catalog,
    load_target_catalog,
)
from codeintel.build.catalogs.hashing import (
    CatalogHashInputs,
    compute_catalog_hash,
    compute_global_catalog_hash,
    compute_hamilton_module_digest,
    compute_schema_registry_hash,
)

__all__ = [
    "CONTRACT_CATALOG_KIND",
    "TARGET_CATALOG_KIND",
    "CatalogHashInputs",
    "compute_catalog_hash",
    "compute_global_catalog_hash",
    "compute_hamilton_module_digest",
    "compute_schema_registry_hash",
    "load_contract_catalog",
    "load_target_catalog",
]
