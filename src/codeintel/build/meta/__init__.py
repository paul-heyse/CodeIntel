"""Build-owned metadata catalog compilation utilities."""

from __future__ import annotations

from codeintel.build.meta.contract_catalog import ContractCatalogResult, persist_contract_catalog

__all__ = [
    "ContractCatalogResult",
    "persist_contract_catalog",
]
